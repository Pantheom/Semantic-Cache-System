# Hybrid Semantic Cache
## System Documentation

> **Version:** 2.0 (Stable)
> **Purpose:** A standalone, production-ready semantic caching layer for LLM applications. Drop it in front of any language model to eliminate redundant API calls for semantically similar queries.

---

## Table of Contents

1. [What Is a Semantic Cache?](#1-what-is-a-semantic-cache)
2. [Why Does It Matter?](#2-why-does-it-matter)
3. [System Architecture](#3-system-architecture)
4. [How a Query Flows Through the System](#4-how-a-query-flows-through-the-system)
5. [The Three-Tier Cache](#5-the-three-tier-cache)
6. [The Privacy Gatekeeper — Query Classifier](#6-the-privacy-gatekeeper--query-classifier)
7. [The Embedding Model](#7-the-embedding-model)
8. [The Database Layer — Supabase](#8-the-database-layer--supabase)
9. [The Bridge API — External Integration Layer](#9-the-bridge-api--external-integration-layer)
10. [The Test Console (Frontend)](#10-the-test-console-frontend)
11. [Seeding the Cache — Datasets](#11-seeding-the-cache--datasets)
12. [Running the System Locally](#12-running-the-system-locally)
13. [Deployment Options](#13-deployment-options)
14. [API Reference](#14-api-reference)
15. [Glossary](#15-glossary)
16. [Version 2 — Scalability](#16-version-2--scalability)

---

## 1. What Is a Semantic Cache?

A **regular cache** remembers exact answers to exact questions. If someone asks *"What is Python?"* and you already answered it, you return the saved answer instead of calling an AI model again.

A **semantic cache** is smarter. It understands the *meaning* of a question, not just the exact words. So if someone asks:

- *"What is Python?"*
- *"Can you explain what Python is?"*
- *"Tell me about the Python programming language"*

...the system recognises all three as the **same question** and returns the cached answer for all of them — even though the words are completely different.

This is the core idea behind this Hybrid Semantic Cache.

---

## 2. Why Does It Matter?

Every time an LLM (like GPT-4, Claude, or Llama) generates a response, it costs:

| Cost | Impact |
|------|--------|
| Money | API calls are billed per token |
| Time | Generation takes 1–10 seconds |
| Energy | GPU compute is energy-intensive |

If many users ask the same general questions (e.g., *"How does photosynthesis work?"*), generating a fresh response every time is wasteful. A semantic cache **answers instantly** from stored responses, saving all three resources.

**Key design principle:** The cache only stores answers to *general* questions. Personal queries (like *"What is my account balance?"*) are detected automatically and never cached — they are flagged as `PERSONAL` and passed back to your application to handle directly.

---

## 3. System Architecture

The system consists of **five components** that work together:

```
+-----------------------------------------------------------+
|                  YOUR APPLICATION                         |
|              (Any LLM-powered backend)                    |
+---------------------------+-------------------------------+
                            | POST /v1/cache/query
                            | X-API-Key: <secret>
                            v
+-----------------------------------------------------------+
|           BRIDGE API  (axiom_bridge.py - Port 8002)       |
|  Auth gate · Request ID · Latency tracking · Error wrap   |
+---------------------------+-------------------------------+
                            | POST /query (internal)
                            v
+-----------------------------------------------------------+
|           MAIN CACHE API  (main.py - Port 8000)           |
|                                                           |
|   +------------+   +-------------+   +--------------+    |
|   | Tier 1     |   | Tier 2      |   | Tier 3       |    |
|   | RAM Cache  |-->| Vector DB   |-->| Cache Miss   |    |
|   | (Exact)    |   | (Semantic)  |   | (no LLM)     |    |
|   +------------+   +-------------+   +------+-------+    |
+-------------------------------------------+--------------+
                                            | POST /classify
                                            v
+-----------------------------------------------------------+
|    CLASSIFIER  (query_classifier.py - Port 8001)          |
|   Layer 1: Heuristic (regex, <1ms)                        |
|   Layer 2: NLI model (DeBERTa, ~10ms)                     |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
|        SUPABASE (PostgreSQL + pgvector)                   |
|        Cloud Vector Database — Always On                  |
+-----------------------------------------------------------+
```

**Components at a glance:**

| Component | File | Port | Role |
|-----------|------|------|------|
| Bridge API | `axiom_bridge.py` | 8002 | External-facing entry point. Auth, request IDs, error enveloping. |
| Main Cache API | `main.py` | 8000 | Core cache engine. Manages all 3 tiers. Internal only. |
| Query Classifier | `query_classifier.py` | 8001 | Decides PERSONAL vs GENERAL. Internal only. |
| Vector Database | Supabase (cloud) | — | Stores embeddings + cached responses persistently. |
| Test Console | `frontend/index.html` | — | Browser UI for testing. |

> **Security note:** Only Port 8002 should be exposed publicly. Ports 8000 and 8001 are internal and should not be accessible from outside the server.

---

## 4. How a Query Flows Through the System

```
Your application sends: "Explain machine learning"
                    |
                    v  POST /v1/cache/query (Port 8002)
         +------------------------------+
         |  Bridge API validates auth   |
         |  Assigns request ID + timer  |
         +---------------------------+--+
                                     | POST /query (internal, Port 8000)
                                     v
             +----------------------------------+
             |  Step 1: RAM Exact Match?        |
             |  Is this exact string in RAM?    |
             +----------------+-----------------+
                    YES <----+----> NO
                     |               |
          Return cached        Step 2: Embed query
          response             (convert to vector)
          + classify                  |
                                      v
                   +-----------------------------------+
                   |  Step 2: DB Semantic Search?      |
                   |  Similar query in Supabase?       |
                   |  (similarity >= 85%)              |
                   +-----------------+-----------------+
                         YES <------+----> NO
                          |                 |
               Return cached          Step 3: Cache Miss
               response               Classify query
               (always GENERAL)       Return null response
               + backfill RAM         + classification
```

**On a HIT:** Your application receives the cached response immediately. No LLM call needed.

**On a MISS:** Your application receives `"cache_hit": false` and a `classification` field:
- `GENERAL` → safe to proceed to your LLM and cache the result externally if you wish
- `PERSONAL` → proceed to your LLM but do not cache the result (user-specific data)

The cache **never calls your LLM**. It is a pure lookup service.

---

## 5. The Three-Tier Cache

### Tier 1 — RAM Exact Match

- **Speed:** < 1 millisecond
- **How it works:** A Python dictionary in memory. If the exact query string was seen before in this process session, the answer is returned instantly.
- **Limit:** Holds up to 10,000 queries. When full, the oldest entry is removed (LRU-style eviction).
- **Scope:** Per-session only. Resets when the server restarts.
- **Classifier:** Runs on RAM hits to return the `classification` field.

### Tier 2 — Database Semantic Match

- **Speed:** 50–200 milliseconds
- **How it works:** The query is converted into a 768-dimension vector using `all-mpnet-base-v2`. That vector is compared against all stored vectors in Supabase using cosine similarity. If a match is found at ≥ 85% similarity, the stored answer is verified by a cross-encoder reranker before being returned.
- **Scope:** Persistent. Shared across all users. Survives restarts.
- **Classifier:** Not called — DB entries are always `GENERAL` by definition (personal queries are never written to the shared DB).

### Tier 2b — Cross-Encoder Reranker

After a vector DB match is found, a second verification step runs:

- **Model:** `BAAI/bge-reranker-base` (cross-encoder)
- **Why:** Vector similarity alone can occasionally match queries that are superficially similar but semantically different. The reranker confirms the match is genuine.
- **Threshold:** 0.70 confidence. Below this, the DB match is rejected and the system falls through to a miss.

### Tier 3 — Cache Miss

- **What happens:** No cached answer was found. The system returns `"cache_hit": false` with a `classification`.
- **Your application** then calls its LLM to generate a response.
- **No LLM call happens inside this service.** The cache is a pure lookup layer.

---

## 6. The Privacy Gatekeeper — Query Classifier

The classifier is the system's privacy guard. It runs on **every request** to classify the query as `GENERAL` or `PERSONAL`.

### Why Is This Needed?

Imagine a user asks: *"What is my bank account balance?"*

If this were stored in the shared cache, future users asking a similar question might receive someone else's private financial data. The classifier prevents this by detecting and flagging personal queries.

### Two Classification Layers

**Layer 1 — Heuristic (Rule-Based) — approximately 0ms**

Hand-crafted rules that catch obvious personal queries instantly:

- Detects PII patterns: email addresses, phone numbers, credit card numbers, SSNs
- Detects personal action phrases: *"remind me"*, *"cancel my"*, *"what is my balance"*, *"show me my orders"*
- Detects possessive + personal noun combos: *"my account"*, *"our records"*, *"my prescription"*

If heuristics classify the query as PERSONAL, the decision is final and Layer 2 is skipped.

**Layer 2 — NLI Model (AI-Based) — approximately 10ms**

If heuristics pass the query as GENERAL, the NLI model does a deeper check:

- **Model:** `cross-encoder/nli-deberta-v3-small`
- **Technique:** Zero-shot classification — no task-specific training needed
- **Internal question:** *"Is this question about the speaker's own private account data, or about general world knowledge?"*
- **Threshold:** 60% confidence → classified as PERSONAL

### Example Classifications

| Query | Layer | Decision | Reason |
|-------|-------|----------|--------|
| "What is my account balance?" | Heuristic | PERSONAL | Possessive + personal noun |
| "Remind me to call the doctor" | Heuristic | PERSONAL | Personal action phrase |
| "john@gmail.com password reset" | Heuristic | PERSONAL | PII pattern detected |
| "How does photosynthesis work?" | NLI | GENERAL | About world knowledge |
| "What causes inflation?" | NLI | GENERAL | About world knowledge |
| "Cancel my subscription" | Heuristic | PERSONAL | Personal action phrase |

### When the Classifier Is Called

| Scenario | Classifier called? | Notes |
|---|---|---|
| RAM Hit | Yes | Returns classification alongside cached response |
| DB Hit | No | DB entries are always GENERAL — classifier skipped for speed |
| Cache Miss | Yes | Returns classification so your app knows the query type |

---

## 7. The Embedding Model

**Model:** `all-mpnet-base-v2` (by Sentence Transformers)

An embedding model converts text into a list of numbers (a "vector") that captures its meaning. Two sentences with similar meanings produce vectors that are mathematically close to each other — even if the words are different.

| Property | Value |
|----------|-------|
| Output dimensions | 768 |
| Max input length | 514 tokens |
| Model size | ~420 MB |
| GPU support | Yes (CUDA) |
| Similarity metric | Cosine similarity |

**How similarity is measured:**

| Score | Meaning |
|-------|---------|
| 1.00 | Identical meaning |
| 0.85+ | Very similar — cache hit threshold |
| 0.50 | Loosely related |
| 0.00 | Completely unrelated |

---

## 8. The Database Layer — Supabase

**Supabase** is a cloud database built on PostgreSQL. The system uses its `pgvector` extension to store and search vector embeddings at scale.

### Table Structure: `shared_llm_cache`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Unique row identifier |
| `query_text` | TEXT | The original question text |
| `response_text` | TEXT | The cached AI response |
| `embedding` | VECTOR(768) | Mathematical representation of the query |
| `created_at` | TIMESTAMPTZ | When this entry was added |
| `hit_count` | INTEGER | How many times this row has been served (V2) |
| `last_accessed_at` | TIMESTAMPTZ | Last time this row was a cache hit (V2) |

### How Vector Search Works

The database function `match_shared_cache` performs an approximate nearest-neighbour search using the query embedding. It returns the closest matching entry above the similarity threshold in milliseconds, even with millions of rows.

---

## 9. The Bridge API — External Integration Layer

`axiom_bridge.py` is the **public-facing entry point** to the semantic cache. It runs on Port 8002 and is the only port that should be exposed to your application.

**What the bridge adds on top of the core cache:**

- **API key authentication** — `X-API-Key` header on every request
- **Request IDs** — every response includes a UUID for end-to-end tracing
- **Latency tracking** — `latency_ms` on every response
- **Structured error envelopes** — consistent error format, never raw exceptions
- **Versioned API surface** — `/v1/` prefix for clean evolution

**Endpoints:**

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | No | Liveness probe (AWS health checks, uptime monitors) |
| `POST` | `/v1/cache/query` | Yes | Look up a query in the cache |
| `GET` | `/v1/stats` | Yes | Session hit/miss statistics |

See [Section 14](#14-api-reference) for full request/response schemas.

---

## 10. The Test Console (Frontend)

The system includes a browser-based test console (`frontend/index.html`) for testing and demonstration.

**Features:**
- Send queries directly to the Bridge API
- See colour-coded responses by source (RAM / DB / Miss)
- View live session statistics (total queries, RAM hits, DB hits, misses)
- Real-time cache hit rate bar
- API online/offline status indicator

**How to use:** Open `frontend/index.html` in a browser while all services are running. No server needed — it is a static HTML file that calls the Bridge API directly.

---

## 11. Seeding the Cache — Datasets

"Seeding" means pre-loading the database with high-quality question-answer pairs before any real users arrive. A well-seeded cache delivers useful hit rates from day one.

### How the Seeding Script Works (`seed_cache.py`)

1. **Streams** the dataset file item-by-item using `ijson` (no full file loaded into RAM — avoids memory crashes on large files)
2. **Filters** out queries longer than 500 characters
3. **Deduplicates** by checking against existing entries in the database
4. **Accumulates** valid records into a batch of 2,000
5. **Batch-embeds** the entire batch at once on GPU (much faster than one-by-one)
6. **Batch-inserts** all 2,000 records to Supabase in a single API call
7. **Saves a checkpoint** after each batch (resumable if interrupted)
8. **Repeats** until the target count is reached

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SEED_COUNT` | 8,000 | Total rows to insert this run |
| `BATCH_SIZE` | 2,000 | Records per embedding + insert cycle |
| `MAX_QUERY_CHARS` | 500 | Skip queries longer than this |

### Recommended Datasets

#### OpenHermes 2.5 — Primary Dataset

- **Source:** `teknium/OpenHermes-2.5` on Hugging Face
- **Size:** ~1 million conversations
- **Content:** Coding, reasoning, science, general knowledge, creative writing
- **Target rows:** 8,000–15,000
- **Status:** Recommended primary dataset

#### Dolly 15K (by Databricks)

- **Source:** `databricks/databricks-dolly-15k` on Hugging Face
- **Size:** 15,000 examples
- **Content:** Open QA, classification, summarisation, general knowledge
- **Why it's great:** Human-curated, very clean, accurate responses
- **Target rows:** 5,000–10,000

#### ShareGPT (Cleaned)

- **Source:** `anon8231489123/ShareGPT_Vicuna_unfiltered` on Hugging Face
- **Size:** ~90,000 conversations
- **Content:** Real user conversations — covers enormous topic diversity
- **Caution:** Pre-filter personal-sounding queries before seeding
- **Target rows:** 3,000–5,000

#### WizardLM Evol-Instruct

- **Source:** `WizardLM/WizardLM_evol_instruct_70k` on Hugging Face
- **Size:** 70,000 examples
- **Content:** Step-by-step reasoning, coding, problem solving
- **Target rows:** 3,000–5,000

### Recommended Seeding Strategy

```
Phase 1 — OpenHermes 2.5:    8,000 rows   (broad coverage)
Phase 2 — Dolly 15K:         5,000 rows   (factual, clean)
Phase 3 — ShareGPT Cleaned:  3,000 rows   (real-world diversity)
-----------------------------------------
Total:                       16,000 rows
```

16,000 high-quality rows provides strong cache hit rates for general Q&A use cases while keeping the database lean and fast.

---

## 12. Running the System Locally

### Prerequisites

- Python 3.10+
- A Supabase project with the `shared_llm_cache` table and `pgvector` enabled
- NVIDIA GPU (optional, but recommended for faster embedding)

### Installation

```bash
# Clone and enter the repository
cd semantic-cache

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

BRIDGE_API_KEY=your-strong-random-secret
CACHE_API_URL=http://127.0.0.1:8000
BRIDGE_PORT=8002
UPSTREAM_TIMEOUT_S=30.0
```

### Starting the Services

Open **three** terminal windows and run one command in each:

**Terminal 1 — Main Cache API (Port 8000):**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Classifier Service (Port 8001):**
```bash
uvicorn query_classifier:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 3 — Bridge API (Port 8002):**
```bash
uvicorn axiom_bridge:app --host 0.0.0.0 --port 8002 --reload
```

### Seeding the Database

```bash
python seed_cache.py
```

Expected output per batch:
```
[BATCH] Embedding 2000 queries on CUDA...
[BATCH] Inserting 2000 records to Supabase...
[BATCH] Inserted 2000 records. Total so far: 2000/8000
[CHECKPOINT] Mid-run checkpoint saved
```

### Running Tests

```bash
pytest tests/test_bridge_api.py -v
```

### Quick Test

```bash
curl -X POST http://localhost:8002/v1/cache/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-strong-random-secret" \
  -d '{"prompt": "What is machine learning?"}'
```

---

## 13. Deployment Options

### Option A — Self-Host + Cloudflare Tunnel (Recommended for Demos)

Run the services on your own machine and expose Port 8002 publicly using Cloudflare Tunnel — completely free.

**Steps:**
1. Install `cloudflared` from cloudflare.com
2. Start all three services
3. Expose the bridge: `cloudflared tunnel --url http://localhost:8002`
4. You receive a permanent public HTTPS URL

**Pros:** Full GPU access, no RAM limits, no cold starts, zero cost
**Cons:** Machine must stay on; not suitable for 24/7 production

### Option B — AWS EC2 (Production)

Run all three services on an EC2 instance. Use the included `deploy_aws.sh` to start and manage the services in the background.

**Steps:**
1. Launch an EC2 instance (Ubuntu 22.04 recommended)
2. Open port `8002/tcp` inbound in the Security Group
3. Clone the repository and install dependencies
4. Add credentials to `.env`
5. Run `./deploy_aws.sh` to start all three services (Cache, Classifier, Bridge)

**Checking service status:**
```bash
tail -f logs/bridge_api.log   # live logs for Port 8002
tail -f logs/main_api.log     # live logs for Port 8000
```

### Option C — Oracle Cloud Always Free

Oracle Cloud's Always Free tier provides a real cloud server at no cost.

**Specs available for free:**
- Up to 4 ARM vCPUs
- Up to 24 GB RAM

24 GB RAM easily handles both the embedding model (~420 MB) and the NLI classifier (~500 MB) with headroom.

### Deployment Comparison

| Option | Cost | Always On | RAM | GPU | Best For |
|--------|------|-----------|-----|-----|----------|
| Self-Host + Cloudflare | Free | While machine is on | Your machine's RAM | Your GPU | Demos, hackathons |
| AWS EC2 | Paid | Yes | Configurable | Optional | Production |
| Oracle Cloud Always Free | Free | Yes | 24 GB | CPU only | Long-term, production |

> **Important:** Do not expose ports 8000 or 8001 publicly. Only Port 8002 (the Bridge API) should accept external traffic.

---

## 14. API Reference

All external requests go to the **Bridge API (Port 8002)**. Ports 8000 and 8001 are internal.

---

### GET /health

Liveness and readiness probe. No authentication required.

**Response:**
```json
{
  "status": "ok",
  "service": "axiom_bridge",
  "version": "1.0.0",
  "upstream_cache": "ok"
}
```

`upstream_cache` reports whether Port 8000 is reachable. Use this for load balancer health checks.

---

### POST /v1/cache/query

The main endpoint. Performs a full three-tier cache lookup.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-bridge-api-key
```

**Request body:**
```json
{
  "prompt": "What is machine learning?",
  "user_id": "optional-user-uuid",
  "session_id": "optional-session-id"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | The query text (1–4096 characters) |
| `user_id` | string | No | Passed through for logging only |
| `session_id` | string | No | Passed through for logging only |

---

**Response — Cache HIT (RAM):**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "cache_hit": true,
  "source": "RAM_Exact_Hit",
  "response": "Machine learning is a subset of AI...",
  "classification": "GENERAL",
  "latency_ms": 8.2,
  "debug": {
    "tier": "RAM",
    "similarity_score": null,
    "reranker_score": null,
    "hit_count": null,
    "classifier_called": true
  }
}
```

**Response — Cache HIT (Semantic DB):**
```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "cache_hit": true,
  "source": "DB_Semantic_Hit",
  "response": "Machine learning is a branch of AI...",
  "classification": "GENERAL",
  "latency_ms": 124.5,
  "debug": {
    "tier": "DB",
    "similarity_score": 0.9124,
    "reranker_score": 0.8312,
    "hit_count": 14,
    "classifier_called": false
  }
}
```

**Response — Cache MISS:**
```json
{
  "request_id": "770e8400-e29b-41d4-a716-446655440002",
  "cache_hit": false,
  "source": "Cache_Miss",
  "response": null,
  "classification": "GENERAL",
  "latency_ms": 87.3,
  "debug": {
    "tier": "MISS",
    "similarity_score": null,
    "reranker_score": null,
    "hit_count": null,
    "classifier_called": true
  }
}
```

**Response field reference:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string (UUID) | Unique ID for this request — use for logging |
| `cache_hit` | boolean | `true` if a cached response was returned |
| `source` | string | `RAM_Exact_Hit` / `DB_Semantic_Hit` / `Cache_Miss` |
| `response` | string / null | Cached answer text, or `null` on a miss |
| `classification` | string | `GENERAL` or `PERSONAL` |
| `latency_ms` | float | Bridge round-trip time in milliseconds |
| `debug.tier` | string | `RAM` / `DB` / `MISS` |
| `debug.similarity_score` | float / null | Cosine similarity (DB hits only) |
| `debug.reranker_score` | float / null | Cross-encoder confidence (DB hits only) |
| `debug.hit_count` | int / null | Lifetime hits for this cached entry (DB hits only) |
| `debug.classifier_called` | boolean | Whether the Privacy Classifier ran |

---

**What `classification` means for your application:**

| Value | On a HIT | On a MISS |
|-------|----------|-----------|
| `GENERAL` | Query is about world knowledge | Safe to call your LLM and cache the result if you wish |
| `PERSONAL` | Informational — entry existed | Call your LLM but treat the result as private (do not cache) |

---

**Error responses:**

```json
{
  "request_id": "...",
  "status": "error",
  "error_code": "INVALID_API_KEY",
  "message": "Missing or invalid X-API-Key header.",
  "latency_ms": 0.5
}
```

| Status Code | Error Code | Meaning |
|-------------|------------|---------|
| 401 | `INVALID_API_KEY` | Missing or wrong `X-API-Key` header |
| 422 | — | Request body validation failed (e.g., empty prompt) |
| 502 | `UPSTREAM_TIMEOUT` | Port 8000 did not respond in time |
| 502 | `UPSTREAM_UNAVAILABLE` | Port 8000 is unreachable |
| 500 | `INTERNAL_ERROR` | Unexpected server error |

---

### GET /v1/stats

Session-level cache performance statistics. Resets on process restart.

**Headers:**
```
X-API-Key: your-bridge-api-key
```

**Response:**
```json
{
  "service": "axiom_bridge",
  "requests_served": 1042,
  "cache_hits": 891,
  "cache_misses": 151,
  "hit_rate_pct": 85.51,
  "avg_latency_ms": 42.7
}
```

---

### Internal Endpoints (Not for External Use)

These are used internally between services. Do not call them from your application.

#### Main Cache API — Port 8000

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Cache API liveness check |
| `POST` | `/query` | Raw cache lookup (called by Bridge API) |

**POST /query request:**
```json
{ "prompt": "What is machine learning?" }
```

**POST /query response:**
```json
{
  "status": "success",
  "source": "RAM_Exact_Hit | DB_Semantic_Hit | Cache_Miss",
  "response": "<cached text or null>",
  "classification": "GENERAL | PERSONAL",
  "debug": { ... }
}
```

#### Classifier Service — Port 8001

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Classifier liveness check |
| `POST` | `/classify` | Full two-layer classification |
| `POST` | `/classify/heuristic` | Heuristic layer only |
| `POST` | `/classify/nli` | NLI model only |
| `POST` | `/classify/compare` | Both layers, results side-by-side |

---

## 15. Glossary

| Term | Meaning |
|------|---------| 
| **Semantic Cache** | A cache that matches queries by meaning, not just exact text |
| **Embedding** | Converting text into a list of numbers representing its meaning |
| **Vector** | A list of numbers representing text in mathematical space |
| **Cosine Similarity** | How similar two vectors are (0 = unrelated, 1 = identical) |
| **pgvector** | A PostgreSQL extension for fast vector similarity search |
| **NLI** | Natural Language Inference — AI reasoning about text relationships |
| **Zero-shot Classification** | Classifying text into categories without task-specific training |
| **DeBERTa** | A transformer model from Microsoft used for NLI tasks |
| **Heuristic** | A rule-based approach using patterns instead of AI |
| **Cross-Encoder Reranker** | A model that verifies a vector similarity match is semantically genuine |
| **Cold Start** | The delay when a service wakes up from sleep and reloads its models |
| **RAM Backfill** | When a DB cache hit is also stored in RAM for faster access next time |
| **Checkpoint** | A saved state allowing a long-running process to resume if interrupted |
| **Supabase** | A cloud database platform built on PostgreSQL |
| **Bridge API** | The external-facing layer that adds auth, request IDs, and error handling |
| **SEED_COUNT** | Target number of rows to insert during a seeding run |
| **BATCH_SIZE** | How many records are embedded and inserted at once |
| **GENERAL** | A query about world knowledge — safe to share across users |
| **PERSONAL** | A query about user-specific data — never stored in the shared cache |

---

## 16. Version 2 — Scalability

> **Status:** ✅ Implemented. Files: `v2_migration.sql`, `.github/workflows/evict_cache.yml`, `main.py`.

### The Problem

The cache is a self-growing database — every new general query gets stored permanently. Over time:
1. **Unbounded growth** — the database grows indefinitely
2. **Stale data** — old, rarely-used entries accumulate and slow down vector search

### The Solution — LFU Eviction via GitHub Actions

A **Least Frequently Used (LFU)** eviction policy automatically removes low-value entries when the database approaches capacity. The eviction runs as a scheduled cron job on **GitHub Actions** — cloud-hosted, free, requires no running server.

### Capacity Design

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Max DB rows | 50,000 | Hard ceiling (safe for Supabase free tier ~500 MB) |
| Eviction threshold | 80% = 40,000 rows | Eviction triggers at this level |
| Eviction target | 70% = 35,000 rows | DB is pruned to this level |
| Grace period | 7 days | New rows cannot be evicted |
| Recency window | 30 days | Recent access counts double in scoring |
| Schedule | Every 2 days | GitHub Actions cron |

### LFU Scoring Formula

```
score = lifetime_hit_count
      + lifetime_hit_count  (×2 bonus if accessed within last 30 days)
      + IMMUNE              (if row is less than 7 days old)

Rows with the lowest score are evicted first.
```

### Version Comparison

| Feature | Version 1.0 | Version 2.0 |
|---------|-------------|-------------|
| Three-tier cache | ✅ Yes | ✅ Yes |
| Privacy classifier | ✅ Yes | ✅ Yes |
| Classification on every response | ❌ No | ✅ Yes |
| Bridge API (external entry point) | ❌ No | ✅ Yes |
| API key authentication | ❌ No | ✅ Yes |
| LLM generation in cache | ✅ Yes (placeholder) | ❌ Removed — pure lookup |
| DB eviction | ❌ No | ✅ Yes (LFU) |
| DB size cap | ❌ No | ✅ 50,000 rows |
| Hit tracking | ❌ No | ✅ Yes |
| Scheduled maintenance | ❌ No | ✅ Yes (GitHub Actions) |

---

*Hybrid Semantic Cache — System Documentation*
*Version 2.0 Stable — August 2026*
