# AXIOM Semantic Cache

A **production-ready semantic caching layer** for LLM-powered applications. Instead of calling an LLM for every query, the system finds semantically similar previously-answered questions and returns the cached response — drastically cutting latency and API costs.

Built for the **AXIOM V2.0** platform as a standalone microservice.

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────┐
│  Tier 1: RAM    │  Exact string match (0ms)
│  (in-memory)    │──────────────────────────► Cache Hit → Return instantly
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│  Tier 2: DB     │  Semantic vector search via pgvector (5–20ms)
│  (Supabase)     │──────────────────────────► Cache Hit → Return + backfill RAM
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│  Privacy Gate   │  Classify: GENERAL or PERSONAL?
│  (Classifier)   │
└────────┬────────┘
         │
    GENERAL → store in DB for future users
    PERSONAL → ephemeral RAM only, never persisted
         │
         ▼
    LLM Generation (your provider)
```

The **Privacy Gatekeeper** (`query_classifier.py`) runs a two-layer check:
- **Layer 1 — Heuristic** (0ms): regex + personal noun patterns
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
git clone https://github.com/your-org/axiom-semantic-cache.git
cd axiom-semantic-cache

python -m venv .venv
.venv\Scripts\activate        # Windows
# or: source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

> **Note for CPU-only machines:** Replace the `torch` line in `requirements.txt` with `torch==2.5.1` (without `+cu121`) before installing.

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your Supabase credentials:

```env
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-anon-key"
```

### 4. Set Up the Supabase Table

Run this SQL in your Supabase SQL Editor:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Cache table
CREATE TABLE shared_llm_cache (
    id            BIGSERIAL PRIMARY KEY,
    query_text    TEXT NOT NULL UNIQUE,
    response_text TEXT NOT NULL,
    embedding     vector(768),
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Semantic search function
CREATE OR REPLACE FUNCTION match_shared_cache(
    query_embedding vector(768),
    match_threshold float,
    match_count     int
)
RETURNS TABLE (
    id            bigint,
    query_text    text,
    response_text text,
    similarity    float
)
LANGUAGE sql STABLE AS $$
    SELECT id, query_text, response_text,
           1 - (embedding <=> query_embedding) AS similarity
    FROM shared_llm_cache
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

### 5. Run the Services

Open **two terminals**:

**Terminal 1 — Classifier (port 8001):**
```bash
uvicorn query_classifier:app --port 8001 --reload
```

**Terminal 2 — Cache API (port 8000):**
```bash
uvicorn main:app --port 8000 --reload
```

### 6. Test It

Open `frontend/index.html` in your browser for the interactive test console, or call the API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

---

## Seeding the Cache

The `seed_cache.py` script populates the database with pre-embedded Q&A pairs for immediate cache hits on common queries.

```bash
python seed_cache.py
```

Edit the config section at the top of `seed_cache.py` to switch datasets. Supported datasets (download locally first):

| Dataset | HuggingFace Path | Format |
|---------|-----------------|--------|
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | `conversations[].value` |
| Dolly 15K | `databricks/databricks-dolly-15k` | `instruction` / `response` |
| WizardLM 70K | `WizardLM/WizardLM_evol_instruct_70k` | `instruction` / `output` |
| FLAN | `Muennighoff/flan` | `inputs` / `targets` |
| ShareGPT | `Aeala/ShareGPT_Vicuna_unfiltered` | `conversations[].value` |

---

## API Reference

### `POST /query`
Main cache lookup endpoint.

**Request:**
```json
{ "prompt": "Explain gradient descent" }
```

**Response:**
```json
{
  "status": "success",
  "source": "DB_Semantic_Hit",
  "response": "Gradient descent is an optimization algorithm...",
  "debug": {
    "tier": "DB",
    "cached": true,
    "similarity_score": 0.9231,
    "classifier_called": false
  }
}
```

**Source values:**
| Source | Meaning |
|--------|---------|
| `RAM_Exact_Hit` | Returned from in-memory cache |
| `DB_Semantic_Hit` | Matched a semantically similar cached query |
| `LLM_Generation_Miss` | Full cache miss — LLM was called |

### `GET /health` (Classifier service)
```json
{ "status": "ok", "nli_model": "cross-encoder/nli-deberta-v3-small", "device": "cuda" }
```

---

## Project Structure

```
semantic_cache/
├── main.py                  # FastAPI cache API (port 8000)
├── query_classifier.py      # Privacy gatekeeper service (port 8001)
├── seed_cache.py            # Dataset seeding script
├── frontend/
│   └── index.html           # Browser-based test console
├── .env.example             # Environment variable template
├── requirements.txt         # Pinned Python dependencies
└── AXIOM_Semantic_Cache_Documentation.md  # Full system documentation
```

---

## Deployment

For a **hackathon / demo environment**, run the services locally and expose them via [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/):

```bash
# Install cloudflared, then:
cloudflared tunnel --url http://localhost:8000
```

This gives you a public HTTPS URL with no port-forwarding or firewall configuration needed.

For **production deployment**, see Section 12 of the [full documentation](./AXIOM_Semantic_Cache_Documentation.md).

---

## Version History

| Version | Status | Notes |
|---------|--------|-------|
| 1.0 | ✅ Stable | Three-tier cache, privacy gatekeeper, ~35K seeded rows |
| 2.0 | 🔜 Planned | LFU eviction, DB size cap (50K rows), GitHub Actions scheduler |

---

## License

MIT — See `LICENSE` for details.
