# Hybrid Semantic Cache

A **production-ready semantic caching layer** for LLM-powered applications. Instead of calling an LLM for every query, the system finds semantically similar previously-answered questions and returns the cached response — cutting latency from seconds to milliseconds and eliminating redundant API costs.

> **Version:** 2.0 Stable — Bridge API, LFU eviction, hit tracking, GitHub Actions scheduler

---

## How It Works

```
Your Application
     │
     │  POST /v1/cache/query  (X-API-Key)
     ▼
┌────────────────────────┐
│  Bridge API (Port 8002) │  Auth, request IDs, error enveloping
└────────────┬───────────┘
             │  internal
             ▼
┌────────────────────────┐
│  Tier 1: RAM           │  Exact string match (<1ms)
│  (in-memory dict)      │───────────────────────────► Hit → return instantly
└────────────┬───────────┘
             │ Miss → embed query
             ▼
┌────────────────────────┐
│  Tier 2: DB            │  Semantic vector search + reranker (50–200ms)
│  (Supabase pgvector)   │───────────────────────────► Hit → return + backfill RAM
└────────────┬───────────┘
             │ Miss
             ▼
┌────────────────────────┐
│  Privacy Classifier    │  GENERAL or PERSONAL?
│  (Port 8001)           │
└────────────┬───────────┘
             │
     GENERAL  →  cache_hit: false, classification: GENERAL
     PERSONAL →  cache_hit: false, classification: PERSONAL
             │
             ▼
     Your application calls its LLM
```

The **Privacy Gatekeeper** (`query_classifier.py`) runs a two-layer check:
- **Layer 1 — Heuristic** (<1ms): regex + personal noun patterns
- **Layer 2 — NLI** (~10ms): DeBERTa-v3-small zero-shot classifier

Only `GENERAL` queries are stored in the shared DB — personal queries are never cached beyond the current session.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- CUDA GPU recommended (CPU works but is slower)
- A [Supabase](https://supabase.com) project with the `pgvector` extension enabled

### 2. Clone & Install

```bash
git clone https://github.com/your-org/semantic-cache.git
cd semantic-cache

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

> **Note for CPU-only machines:** Replace the `torch` line in `requirements.txt` with `torch==2.5.1` (without `+cu121`) before installing.

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-anon-key"

BRIDGE_API_KEY="generate-a-strong-secret"   # your application sends this as X-API-Key
```

### 4. Set Up the Supabase Table

**Fresh install:** Run this SQL in your Supabase SQL Editor:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Cache table
CREATE TABLE shared_llm_cache (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text    TEXT NOT NULL UNIQUE,
    response_text TEXT NOT NULL,
    embedding     vector(768),
    created_at    TIMESTAMPTZ DEFAULT now()
);
```

**Existing install (Version 1 → 2 migration):** Run `v2_migration.sql` in your Supabase SQL Editor to add hit tracking, LFU indexing, and the `evict_lfu_cache()` stored procedure. The script is idempotent — safe to run multiple times.

### 5. Run the Services

Open **three terminals**:

**Terminal 1 — Classifier (Port 8001):**
```bash
uvicorn query_classifier:app --port 8001
```

**Terminal 2 — Cache API (Port 8000):**
```bash
uvicorn main:app --port 8000
```

**Terminal 3 — Bridge API (Port 8002):**
```bash
uvicorn axiom_bridge:app --port 8002
```

### 6. Test It

```bash
# Check the bridge is up
curl http://localhost:8002/health

# Make a cache query
curl -X POST http://localhost:8002/v1/cache/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-bridge-api-key" \
  -d '{"prompt": "What is machine learning?"}'
```

Or open `frontend/index.html` in your browser for the interactive test console.

---

## Integrating with Your Application

See **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** for complete integration examples in Python, JavaScript/TypeScript, and cURL.

**The one-line summary:** Check the cache before every LLM call. If `cache_hit: true`, use the response. If `cache_hit: false`, call your LLM.

```python
result = requests.post(
    "http://your-cache-host:8002/v1/cache/query",
    headers={"X-API-Key": API_KEY},
    json={"prompt": user_message},
).json()

if result["cache_hit"]:
    return result["response"]   # serve from cache
else:
    return call_your_llm(user_message)   # go to LLM
```

---

## Seeding the Cache

The `seed_cache.py` script populates the database with pre-embedded Q&A pairs for immediate cache hits on common queries.

```bash
python seed_cache.py
```

Supported datasets (download locally first):

| Dataset | HuggingFace Path | Recommended rows |
|---------|-----------------|-----------------|
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | 8,000 |
| Dolly 15K | `databricks/databricks-dolly-15k` | 5,000 |
| WizardLM 70K | `WizardLM/WizardLM_evol_instruct_70k` | 3,000 |
| ShareGPT | `Aeala/ShareGPT_Vicuna_unfiltered` | 3,000 |

---

## Bridge API Reference

All external requests go to the **Bridge API (Port 8002)**.

### `POST /v1/cache/query`

**Request:**
```json
{
  "prompt": "Explain gradient descent",
  "user_id": "optional-uuid",
  "session_id": "optional-session-id"
}
```

**Response (hit):**
```json
{
  "request_id": "uuid",
  "cache_hit": true,
  "source": "DB_Semantic_Hit",
  "response": "Gradient descent is an optimization algorithm...",
  "classification": "GENERAL",
  "latency_ms": 145.2,
  "debug": { "tier": "DB", "similarity_score": 0.923, "reranker_score": 0.84, "hit_count": 7 }
}
```

**Response (miss):**
```json
{
  "request_id": "uuid",
  "cache_hit": false,
  "source": "Cache_Miss",
  "response": null,
  "classification": "GENERAL",
  "latency_ms": 87.3,
  "debug": { "tier": "MISS", "classifier_called": true }
}
```

| Source | Meaning |
|--------|---------|
| `RAM_Exact_Hit` | Exact match in session memory — <5ms |
| `DB_Semantic_Hit` | Semantic match in vector database — 50–300ms |
| `Cache_Miss` | No match found — your application should call its LLM |

### `GET /health` — No auth required
### `GET /v1/stats` — Session hit/miss statistics

---

## Project Structure

```
semantic-cache/
├── main.py                    # Cache API (Port 8000) — three-tier lookup engine
├── axiom_bridge.py            # Bridge API (Port 8002) — public-facing entry point
├── bridge_models.py           # Pydantic schemas for the Bridge API
├── query_classifier.py        # Privacy Gatekeeper (Port 8001)
├── seed_cache.py              # Dataset seeding script
├── deploy_aws.sh              # EC2 deployment script (main cache + classifier)
├── deploy_bridge.sh           # EC2 deployment script (bridge API as systemd service)
├── v2_migration.sql           # V2 database migration
├── .github/workflows/
│   └── evict_cache.yml        # Scheduled LFU eviction (GitHub Actions, every 2 days)
├── frontend/
│   └── index.html             # Browser-based test console
├── tests/
│   └── test_bridge_api.py     # Integration tests (14 tests)
├── .env.example               # Environment variable template
├── requirements.txt           # Pinned Python dependencies
├── Semantic_Cache_Documentation.md   # Full system documentation
└── INTEGRATION_GUIDE.md              # Integration guide for LLM applications
```

---

## Deployment

**Demo / Hackathon — Cloudflare Tunnel (free, no setup):**
```bash
cloudflared tunnel --url http://localhost:8002
```
This gives a public HTTPS URL instantly with no port-forwarding needed.

**Production — AWS EC2:**
1. Open port `8002/tcp` inbound in Security Group
2. Add credentials to `.env`
3. Run `./deploy_aws.sh` (starts Ports 8000 + 8001)
4. Run `./deploy_bridge.sh` (starts Port 8002 as systemd service)

See [Semantic_Cache_Documentation.md](./Semantic_Cache_Documentation.md) for full deployment details.

> **Security:** Only expose Port 8002 publicly. Ports 8000 and 8001 are internal.

---

## GitHub Actions Setup (V2 — LFU Eviction)

The database is automatically pruned when it approaches 50,000 rows.

1. Push this repo to GitHub
2. Go to **Settings → Secrets → Actions**
3. Add: `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (the service_role key from Supabase → Settings → API)
4. The workflow at `.github/workflows/evict_cache.yml` fires every 2 days at 02:00 UTC
5. Manual run: **Actions → LFU Cache Eviction → Run workflow**

---

## Running Tests

```bash
pytest tests/test_bridge_api.py -v
```

14 tests covering: health, auth, RAM hits, DB hits, GENERAL misses, PERSONAL misses, stats.

---

## Version History

| Version | Status | Notes |
|---------|--------|-------|
| 1.0 | Stable | Three-tier cache, privacy gatekeeper, seeding pipeline |
| 2.0 | Stable | Bridge API (Port 8002), X-API-Key auth, classification on all responses, LFU eviction, hit tracking, GitHub Actions scheduler |

---

## License

MIT — See `LICENSE` for details.
