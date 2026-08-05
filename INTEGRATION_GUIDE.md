# Semantic Cache — Integration Guide
## How to Connect Any LLM System to the Semantic Cache

This guide shows you how to integrate the Hybrid Semantic Cache into any LLM-powered application. Language examples are included for Python, JavaScript/Node.js, and cURL.

---

## Table of Contents

1. [Overview — What the Cache Does for You](#1-overview--what-the-cache-does-for-you)
2. [Quick Start — 5 Minutes](#2-quick-start--5-minutes)
3. [Understanding the Response](#3-understanding-the-response)
4. [The Full Integration Pattern](#4-the-full-integration-pattern)
5. [Python Integration](#5-python-integration)
6. [JavaScript / Node.js Integration](#6-javascript--nodejs-integration)
7. [cURL Examples](#7-curl-examples)
8. [Handling the Classification Field](#8-handling-the-classification-field)
9. [Handling Errors Gracefully](#9-handling-errors-gracefully)
10. [Performance Tips](#10-performance-tips)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Overview — What the Cache Does for You

Before integrating, understand the one rule that governs everything:

> **The cache is a read-only lookup service. It does not call any LLM. You are always in control of LLM generation.**

Your application's job is simple:

```
1. Ask the cache: "Do you have an answer for this query?"
2. If YES (cache_hit: true)  → use the cached answer. Done. No LLM call needed.
3. If NO  (cache_hit: false) → call your LLM as normal. 
                               Read classification to understand query type.
```

That is the entire integration. The cache sits in front of your LLM and handles the common cases, saving you API costs and reducing latency for your users.

---

## 2. Quick Start — 5 Minutes

### Step 1 — Get your API key

Contact the operator of the Semantic Cache deployment to receive a `BRIDGE_API_KEY`. Store it as an environment variable — never hardcode it.

### Step 2 — Know the endpoint

```
POST https://<your-cache-host>:8002/v1/cache/query
```

### Step 3 — Make your first request

```bash
curl -X POST https://<your-cache-host>:8002/v1/cache/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"prompt": "What is machine learning?"}'
```

### Step 4 — Read the response

```json
{
  "cache_hit": true,
  "source": "DB_Semantic_Hit",
  "response": "Machine learning is a field of AI that allows systems to learn...",
  "classification": "GENERAL",
  "latency_ms": 145.2
}
```

`cache_hit: true` → use `response` directly. Your LLM was not called.

---

## 3. Understanding the Response

Every response from the cache has the same structure:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "cache_hit": true,
  "source": "RAM_Exact_Hit | DB_Semantic_Hit | Cache_Miss",
  "response": "<cached text, or null on a miss>",
  "classification": "GENERAL | PERSONAL",
  "latency_ms": 87.3,
  "debug": {
    "tier": "RAM | DB | MISS",
    "similarity_score": 0.9124,
    "reranker_score": 0.8312,
    "hit_count": 14,
    "classifier_called": true
  }
}
```

### The Fields You Care About

| Field | What to do with it |
|-------|-------------------|
| `cache_hit` | **The main switch.** `true` → skip LLM, use `response`. `false` → call LLM. |
| `response` | The cached answer. Present on hits, `null` on misses. |
| `classification` | `GENERAL` or `PERSONAL`. Tells you the nature of the query. |
| `latency_ms` | Useful for monitoring. Expect 1–300ms depending on the tier. |
| `request_id` | Log this for debugging end-to-end request tracing. |

### What `source` tells you

| Source | Meaning | Typical Latency |
|--------|---------|----------------|
| `RAM_Exact_Hit` | Exact query seen before in this session — returned from memory | < 5ms |
| `DB_Semantic_Hit` | Semantically similar query found in the vector database | 50–300ms |
| `Cache_Miss` | No match found. You must call your LLM. | 50–200ms |

---

## 4. The Full Integration Pattern

Here is the complete decision flow your application should implement:

```
Receive user message
        │
        ▼
POST /v1/cache/query
        │
        ├── cache_hit: true ──────────────────────────────────────────┐
        │                                                             │
        │   classification: GENERAL or PERSONAL (informational)      │
        │                                                             ▼
        │                                                    Use cached response
        │                                                    Return to user ✓
        │
        └── cache_hit: false
                │
                ├── classification: PERSONAL
                │       │
                │       ▼
                │   Call LLM → return to user
                │   Do NOT cache this response anywhere
                │   (user-specific data, privacy-sensitive)
                │
                └── classification: GENERAL
                        │
                        ▼
                    Call LLM → return to user
                    Optionally: update your own cache or DB
                    for future use (outside this service's scope)
```

**Key points:**
- On `PERSONAL` queries: always call your LLM and never cache the result
- On `GENERAL` misses: call your LLM; the response could be cached externally for reuse
- Never block the user while waiting for the cache if it's taking too long — set a timeout and fall back to your LLM directly (see [Handling Errors](#9-handling-errors-gracefully))

---

## 5. Python Integration

### Minimal Example

```python
import os
import requests

CACHE_URL = os.getenv("CACHE_URL", "http://your-cache-host:8002")
CACHE_API_KEY = os.getenv("CACHE_API_KEY")

def query_cache(prompt: str, user_id: str = None) -> dict:
    response = requests.post(
        f"{CACHE_URL}/v1/cache/query",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": CACHE_API_KEY,
        },
        json={"prompt": prompt, "user_id": user_id},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def handle_user_message(user_message: str, user_id: str) -> str:
    cache_result = query_cache(user_message, user_id)

    if cache_result["cache_hit"]:
        print(f"[CACHE HIT] source={cache_result['source']}, latency={cache_result['latency_ms']}ms")
        return cache_result["response"]

    # Cache miss — call your LLM here
    print(f"[CACHE MISS] classification={cache_result['classification']}")
    llm_response = call_your_llm(user_message)
    return llm_response


def call_your_llm(prompt: str) -> str:
    # Replace with your actual LLM call (OpenAI, Groq, Bedrock, Ollama, etc.)
    raise NotImplementedError("Wire in your LLM here")
```

---

### Production Example with Error Handling, Timeout Fallback, and Logging

```python
import os
import logging
import requests
from requests.exceptions import Timeout, RequestException

log = logging.getLogger(__name__)

CACHE_URL     = os.getenv("CACHE_URL", "http://your-cache-host:8002")
CACHE_API_KEY = os.getenv("CACHE_API_KEY")
CACHE_TIMEOUT = float(os.getenv("CACHE_TIMEOUT_S", "5.0"))


def query_semantic_cache(prompt: str, user_id: str = None) -> dict | None:
    """
    Queries the semantic cache. Returns the parsed response dict on success,
    or None if the cache is unavailable (timeout, network error, etc.).

    Never raises — your LLM is always the fallback.
    """
    try:
        resp = requests.post(
            f"{CACHE_URL}/v1/cache/query",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": CACHE_API_KEY,
            },
            json={"prompt": prompt, "user_id": user_id},
            timeout=CACHE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    except Timeout:
        log.warning("[CACHE] Timeout after %.1fs — falling back to LLM", CACHE_TIMEOUT)
        return None

    except RequestException as e:
        log.warning("[CACHE] Unavailable (%s) — falling back to LLM", e)
        return None

    except Exception as e:
        log.error("[CACHE] Unexpected error: %s", e, exc_info=True)
        return None


def generate_response(user_message: str, user_id: str) -> dict:
    """
    Main handler. Returns a dict with the answer and metadata.
    """
    # 1. Try the cache first
    cache_result = query_semantic_cache(user_message, user_id)

    if cache_result is not None and cache_result["cache_hit"]:
        log.info(
            "[CACHE HIT] request_id=%s source=%s latency=%.1fms",
            cache_result.get("request_id"), 
            cache_result["source"], 
            cache_result["latency_ms"],
        )
        return {
            "answer":       cache_result["response"],
            "from_cache":   True,
            "source":       cache_result["source"],
            "latency_ms":   cache_result["latency_ms"],
        }

    # 2. Cache miss (or cache unavailable) — determine if safe to share
    classification = "GENERAL"
    if cache_result is not None:
        classification = cache_result.get("classification", "GENERAL")
        log.info("[CACHE MISS] classification=%s", classification)
    else:
        log.info("[CACHE UNAVAILABLE] Skipping cache, going to LLM")

    # 3. Call your LLM
    llm_answer = call_your_llm(user_message)

    # 4. Log the result
    log.info("[LLM] Generated response. classification=%s", classification)

    return {
        "answer":         llm_answer,
        "from_cache":     False,
        "classification": classification,
        "source":         "LLM",
    }


def call_your_llm(prompt: str) -> str:
    # Example with OpenAI:
    #
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    raise NotImplementedError("Wire in your LLM here")
```

---

### Async Python Example (for FastAPI / asyncio applications)

```python
import os
import httpx
import logging

log = logging.getLogger(__name__)

CACHE_URL     = os.getenv("CACHE_URL", "http://your-cache-host:8002")
CACHE_API_KEY = os.getenv("CACHE_API_KEY")
CACHE_TIMEOUT = float(os.getenv("CACHE_TIMEOUT_S", "5.0"))

# Reuse a single client across requests — much faster than creating per-request
_cache_client: httpx.AsyncClient | None = None

async def get_cache_client() -> httpx.AsyncClient:
    global _cache_client
    if _cache_client is None or _cache_client.is_closed:
        _cache_client = httpx.AsyncClient(
            base_url=CACHE_URL,
            headers={"X-API-Key": CACHE_API_KEY},
            timeout=CACHE_TIMEOUT,
        )
    return _cache_client


async def query_semantic_cache(prompt: str, user_id: str = None) -> dict | None:
    """Async cache lookup. Returns None on any error — LLM is always the fallback."""
    try:
        client = await get_cache_client()
        resp = await client.post(
            "/v1/cache/query",
            json={"prompt": prompt, "user_id": user_id},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        log.warning("[CACHE] Timeout — falling back to LLM")
        return None
    except Exception as e:
        log.warning("[CACHE] Error: %s — falling back to LLM", e)
        return None


async def generate_response(user_message: str, user_id: str = None) -> dict:
    cache_result = await query_semantic_cache(user_message, user_id)

    if cache_result and cache_result["cache_hit"]:
        return {
            "answer": cache_result["response"],
            "from_cache": True,
            "source": cache_result["source"],
        }

    # Miss or unavailable — call LLM
    llm_answer = await call_your_llm_async(user_message)
    return {"answer": llm_answer, "from_cache": False, "source": "LLM"}


async def call_your_llm_async(prompt: str) -> str:
    raise NotImplementedError("Wire in your async LLM call here")
```

---

## 6. JavaScript / Node.js Integration

### Minimal Example

```javascript
const CACHE_URL     = process.env.CACHE_URL || 'http://your-cache-host:8002';
const CACHE_API_KEY = process.env.CACHE_API_KEY;

async function queryCache(prompt, userId = null) {
  const response = await fetch(`${CACHE_URL}/v1/cache/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': CACHE_API_KEY,
    },
    body: JSON.stringify({ prompt, user_id: userId }),
    signal: AbortSignal.timeout(5000),  // 5-second timeout
  });

  if (!response.ok) {
    throw new Error(`Cache returned HTTP ${response.status}`);
  }

  return response.json();
}


async function handleUserMessage(userMessage, userId) {
  let cacheResult = null;

  try {
    cacheResult = await queryCache(userMessage, userId);
  } catch (err) {
    console.warn('[CACHE] Unavailable, falling back to LLM:', err.message);
  }

  if (cacheResult?.cache_hit) {
    console.log(`[CACHE HIT] source=${cacheResult.source} latency=${cacheResult.latency_ms}ms`);
    return cacheResult.response;
  }

  const classification = cacheResult?.classification ?? 'GENERAL';
  console.log(`[CACHE MISS] classification=${classification}`);

  // Call your LLM here
  const llmResponse = await callYourLLM(userMessage);
  return llmResponse;
}


async function callYourLLM(prompt) {
  // Example with OpenAI Node SDK:
  //
  // const { OpenAI } = require('openai');
  // const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  // const completion = await client.chat.completions.create({
  //   model: 'gpt-4o',
  //   messages: [{ role: 'user', content: prompt }],
  // });
  // return completion.choices[0].message.content;
  throw new Error('Wire in your LLM here');
}
```

---

### TypeScript Example (Next.js / Express)

```typescript
interface CacheResponse {
  request_id: string;
  cache_hit: boolean;
  source: 'RAM_Exact_Hit' | 'DB_Semantic_Hit' | 'Cache_Miss';
  response: string | null;
  classification: 'GENERAL' | 'PERSONAL';
  latency_ms: number;
  debug: {
    tier: string;
    similarity_score: number | null;
    reranker_score: number | null;
    hit_count: number | null;
    classifier_called: boolean;
  };
}

interface MessageResult {
  answer: string;
  fromCache: boolean;
  source: string;
  classification?: string;
}

const CACHE_URL     = process.env.CACHE_URL!;
const CACHE_API_KEY = process.env.CACHE_API_KEY!;

async function querySemanticCache(
  prompt: string,
  userId?: string
): Promise<CacheResponse | null> {
  try {
    const response = await fetch(`${CACHE_URL}/v1/cache/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': CACHE_API_KEY,
      },
      body: JSON.stringify({ prompt, user_id: userId ?? null }),
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      console.warn(`[CACHE] HTTP ${response.status}`);
      return null;
    }

    return (await response.json()) as CacheResponse;
  } catch (err) {
    console.warn('[CACHE] Unavailable:', err);
    return null;
  }
}


export async function generateResponse(
  userMessage: string,
  userId?: string
): Promise<MessageResult> {
  const cacheResult = await querySemanticCache(userMessage, userId);

  if (cacheResult?.cache_hit && cacheResult.response) {
    return {
      answer: cacheResult.response,
      fromCache: true,
      source: cacheResult.source,
      classification: cacheResult.classification,
    };
  }

  // Miss or cache unavailable — call your LLM
  const answer = await callYourLLM(userMessage);
  return {
    answer,
    fromCache: false,
    source: 'LLM',
    classification: cacheResult?.classification,
  };
}

async function callYourLLM(prompt: string): Promise<string> {
  throw new Error('Wire in your LLM here');
}
```

---

## 7. cURL Examples

### Check if the cache is online

```bash
curl https://<your-cache-host>:8002/health
```

Expected response:
```json
{"status": "ok", "service": "axiom_bridge", "version": "1.0.0", "upstream_cache": "ok"}
```

---

### Query the cache (cache hit example)

```bash
curl -X POST https://<your-cache-host>:8002/v1/cache/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"prompt": "What is machine learning?"}'
```

---

### Query with optional fields

```bash
curl -X POST https://<your-cache-host>:8002/v1/cache/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "prompt": "Explain photosynthesis simply",
    "user_id": "user-uuid-1234",
    "session_id": "session-abc-xyz"
  }'
```

---

### Test a personal query (should return classification: PERSONAL)

```bash
curl -X POST https://<your-cache-host>:8002/v1/cache/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"prompt": "What is my account balance?"}'
```

Expected result: `cache_hit: false`, `classification: "PERSONAL"` — even if this question was asked before, it is never stored in the shared cache.

---

### Get session statistics

```bash
curl https://<your-cache-host>:8002/v1/stats \
  -H "X-API-Key: YOUR_API_KEY"
```

---

## 8. Handling the Classification Field

The `classification` field tells you whether the query was about general world knowledge (`GENERAL`) or about something user-specific or private (`PERSONAL`).

### What GENERAL means

The query is about general knowledge:
- *"What is machine learning?"*
- *"How does photosynthesis work?"*
- *"Explain REST APIs"*
- *"What is the capital of France?"*

**On a HIT with GENERAL:** you received a cached shared answer — correct and expected.

**On a MISS with GENERAL:** the cache didn't have this yet. You call your LLM. The answer is shareable — it could theoretically be added to the cache for future users.

### What PERSONAL means

The query is about something specific to a user:
- *"What is my account balance?"*
- *"Show me my order history"*
- *"Cancel my subscription"*
- *"Remind me to call the doctor"*

**On a MISS with PERSONAL:** the cache intentionally has no answer (personal data is never stored in the shared cache). You call your LLM with the user's real context. **Do not add this response to any shared cache.**

### Decision table

```
cache_hit  | classification | What to do
-----------|----------------|--------------------------------------------
true       | GENERAL        | Use cached response. No LLM call needed. ✓
true       | PERSONAL*      | Use cached response. No LLM call needed. ✓
false      | GENERAL        | Call LLM. Response could be cached if desired.
false      | PERSONAL       | Call LLM with user context. Never cache result.

* PERSONAL hits are rare — only possible from RAM (session memory).
  DB hits are always GENERAL by design.
```

### Example — routing based on classification

```python
def handle_message(user_message: str, user_id: str, user_context: dict) -> str:
    result = query_semantic_cache(user_message)

    # Hit — always use the cached response
    if result and result["cache_hit"]:
        return result["response"]

    classification = result["classification"] if result else "GENERAL"

    if classification == "PERSONAL":
        # Personal query — inject user context into your LLM prompt
        system_prompt = build_personal_prompt(user_context)
        return call_your_llm(user_message, system_prompt=system_prompt)
    else:
        # General query — standard LLM call
        return call_your_llm(user_message)
```

---

## 9. Handling Errors Gracefully

The semantic cache should **never block your application**. Always implement a fallback:

### The Golden Rule

```
Cache unavailable → fall through to LLM, log the issue, continue.
```

Never let a cache failure break the user experience. The cache is an optimisation, not a dependency.

### Error Types and Responses

| HTTP Status | Meaning | Your action |
|-------------|---------|-------------|
| `200` with `cache_hit: false` | Normal miss — no error | Call your LLM |
| `401` | Wrong or missing API key | Fix your `X-API-Key` header |
| `422` | Bad request body | Fix your JSON (likely empty prompt) |
| `502` | Cache internal service is down | Fallback to LLM, alert your ops team |
| `500` | Unexpected server error | Fallback to LLM, log the `request_id` |
| Timeout / connection refused | Cache process is unreachable | Fallback to LLM |

### Python Error Handling Pattern

```python
def safe_cache_query(prompt: str) -> dict | None:
    """
    Returns the cache result or None.
    None means "skip cache, go to LLM" — not an error your user sees.
    """
    try:
        resp = requests.post(
            f"{CACHE_URL}/v1/cache/query",
            headers={"Content-Type": "application/json", "X-API-Key": CACHE_API_KEY},
            json={"prompt": prompt},
            timeout=3.0,   # Short timeout — don't make users wait for a slow cache
        )
        if resp.status_code == 401:
            log.error("[CACHE] Auth failed — check CACHE_API_KEY")
            return None
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        log.warning("[CACHE] Timeout")
        return None
    except requests.exceptions.ConnectionError:
        log.warning("[CACHE] Connection refused — is the cache running?")
        return None
    except Exception as e:
        log.error("[CACHE] Unexpected: %s", e)
        return None
```

### JavaScript Error Handling Pattern

```javascript
async function safeCacheQuery(prompt) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000); // 3s timeout

    const response = await fetch(`${CACHE_URL}/v1/cache/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': CACHE_API_KEY,
      },
      body: JSON.stringify({ prompt }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (response.status === 401) {
      console.error('[CACHE] Auth failed — check CACHE_API_KEY');
      return null;
    }

    if (!response.ok) {
      console.warn(`[CACHE] HTTP ${response.status}`);
      return null;
    }

    return await response.json();

  } catch (err) {
    if (err.name === 'AbortError') {
      console.warn('[CACHE] Request timed out');
    } else {
      console.warn('[CACHE] Error:', err.message);
    }
    return null;
  }
}
```

---

## 10. Performance Tips

### 1. Set a short timeout on your cache calls

The cache should respond in under 300ms. If it takes longer, something is wrong. Don't make your users wait — fall back to your LLM quickly.

```python
timeout=3.0   # 3 seconds max — generous but firm
```

### 2. Reuse the HTTP client

Creating a new HTTP connection for every request adds 20–100ms of overhead. Use a persistent client:

```python
# Python — create once, reuse across requests
session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "X-API-Key": CACHE_API_KEY,
})

# All requests use the same TCP connection pool
result = session.post(f"{CACHE_URL}/v1/cache/query", json={"prompt": ...}).json()
```

```javascript
// Node.js — use a persistent agent
const https = require('https');
const agent = new https.Agent({ keepAlive: true, maxSockets: 10 });
// Pass agent: agent to your fetch options
```

### 3. Send `user_id` and `session_id`

These fields are optional but help with debugging. When an issue occurs, search your logs for the `request_id` and correlate with `user_id` and `session_id` to trace exactly what happened.

### 4. Log `request_id`

Every cache response includes a `request_id` UUID. Log it alongside your own request context:

```python
log.info("[REQUEST] user=%s cache_request_id=%s hit=%s",
         user_id, result["request_id"], result["cache_hit"])
```

### 5. Don't call the cache for system/meta queries

If your application makes internal calls (health checks, admin operations, tool calls that aren't user queries), skip the cache entirely for those. Only call the cache for natural language user queries that could realistically have been answered before.

### 6. Monitor the hit rate via /v1/stats

Periodically poll `/v1/stats` to understand how well the cache is working:

```python
def get_cache_stats() -> dict:
    resp = requests.get(
        f"{CACHE_URL}/v1/stats",
        headers={"X-API-Key": CACHE_API_KEY},
        timeout=3.0,
    )
    return resp.json()

stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate_pct']}%")
print(f"Avg latency: {stats['avg_latency_ms']}ms")
```

A healthy, well-seeded cache should achieve 40–70% hit rate on general Q&A workloads after warm-up.

---

## 11. Configuration Reference

### Environment Variables (on your application side)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CACHE_URL` | Yes | — | Full URL of the Bridge API, e.g. `http://host:8002` |
| `CACHE_API_KEY` | Yes | — | The shared secret `X-API-Key` value |
| `CACHE_TIMEOUT_S` | No | `5.0` | Max seconds to wait for a cache response |

### Bridge API Configuration (on the cache server side)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BRIDGE_API_KEY` | Yes | — | The secret your application sends as `X-API-Key` |
| `CACHE_API_URL` | No | `http://127.0.0.1:8000` | Internal address of Port 8000 |
| `BRIDGE_PORT` | No | `8002` | Port the bridge listens on |
| `UPSTREAM_TIMEOUT_S` | No | `30.0` | Max seconds to wait for Port 8000 |

---

## Appendix — Response Shape Quick Reference

**HIT:**
```json
{
  "request_id": "uuid",
  "cache_hit": true,
  "source": "RAM_Exact_Hit | DB_Semantic_Hit",
  "response": "The cached answer...",
  "classification": "GENERAL | PERSONAL",
  "latency_ms": 45.2,
  "debug": { "tier": "RAM | DB", "similarity_score": 0.91, "reranker_score": 0.83, "hit_count": 7, "classifier_called": true }
}
```

**MISS:**
```json
{
  "request_id": "uuid",
  "cache_hit": false,
  "source": "Cache_Miss",
  "response": null,
  "classification": "GENERAL | PERSONAL",
  "latency_ms": 110.5,
  "debug": { "tier": "MISS", "similarity_score": null, "reranker_score": null, "hit_count": null, "classifier_called": true }
}
```

**ERROR (401):**
```json
{
  "request_id": "uuid",
  "status": "error",
  "error_code": "INVALID_API_KEY",
  "message": "Missing or invalid X-API-Key header.",
  "latency_ms": 0.3
}
```

---

*Hybrid Semantic Cache — Integration Guide*
*Version 2.0 — August 2026*
