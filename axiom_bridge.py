"""
bridge_api.py — Semantic Cache Bridge API

Public-facing REST API that sits in front of the internal Semantic Cache (Port 8000).
It is the ONLY entry point external applications use to interact with the cache.

Port  8002 — this file (Bridge API, external-facing)
Port  8000 — main.py (Cache API, internal)
Port  8001 — query_classifier.py (Classifier, internal)

    Your Application
        │
        │  POST /v1/cache/query  (X-API-Key required)
        ▼
    Bridge API (Port 8002)       ← This file
        │
        │  POST /query           (internal)
        ▼
    Cache API (Port 8000)        ← main.py
        │
        │  POST /classify        (on RAM hits and misses)
        ▼
    Classifier (Port 8001)       ← query_classifier.py

On HIT  → bridge returns cached response to your application.
On MISS → bridge returns cache_hit=false + classification.
          Your application then calls its LLM directly.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bridge_models import (
    CacheQueryRequest,
    CacheQueryResponse,
    CacheDebugInfo,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

BRIDGE_API_KEY: str = os.getenv("BRIDGE_API_KEY", "")
CACHE_API_URL:  str = os.getenv("CACHE_API_URL", "http://127.0.0.1:8000")
BRIDGE_PORT:    int = int(os.getenv("BRIDGE_PORT", "8002"))

# Timeout for upstream calls to Port 8000 (seconds).
# Tier 3 (LLM generation) can be slow — set generously.
UPSTREAM_TIMEOUT: float = float(os.getenv("UPSTREAM_TIMEOUT_S", "30.0"))

if not BRIDGE_API_KEY:
    raise RuntimeError(
        "[BRIDGE] BRIDGE_API_KEY is not set. "
        "Add it to your .env file before starting this service."
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bridge_api")

# ---------------------------------------------------------------------------
# Session-level stats (in-memory, resets on process restart)
# ---------------------------------------------------------------------------

_stats = {
    "requests_served": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_latency_ms": 0.0,
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage startup and shutdown of the shared HTTP client."""
    global _http_client
    _http_client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    log.info("[BRIDGE] Starting on port %d", BRIDGE_PORT)
    log.info("[BRIDGE] Upstream cache API: %s", CACHE_API_URL)
    log.info("[BRIDGE] API key auth: ENABLED")
    yield
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    log.info("[BRIDGE] Shutdown complete.")


app = FastAPI(
    title="Semantic Cache Bridge API",
    description=(
        "Public-facing integration layer between your LLM application "
        "and the internal Hybrid Semantic Cache. "
        "On a MISS: your application should proceed to its LLM. "
        "Read `classification` to determine whether the query is GENERAL or PERSONAL."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth dependency — X-API-Key header validation
# ---------------------------------------------------------------------------

async def verify_api_key(request: Request) -> None:
    """
    Validates the X-API-Key header on every request.
    Returns 401 if the key is missing or does not match BRIDGE_API_KEY.
    Called via FastAPI's Depends() — applied to every protected endpoint.
    """
    api_key: Optional[str] = request.headers.get("X-API-Key")
    if not api_key or api_key != BRIDGE_API_KEY:
        log.warning(
            "[AUTH] Rejected request from %s — missing or invalid X-API-Key",
            request.client.host if request.client else "unknown"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "error_code": "INVALID_API_KEY",
                "message": (
                    "Missing or invalid X-API-Key header. "
                    "Include your shared secret as X-API-Key: <secret>."
                ),
            },
        )

# ---------------------------------------------------------------------------
# Shared async HTTP client (connection-pooled, reused across requests)
# ---------------------------------------------------------------------------

_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    return _http_client


# Startup/shutdown is handled by the lifespan context manager above.

# ---------------------------------------------------------------------------
# Helper — call upstream Cache API (Port 8000)
# ---------------------------------------------------------------------------

async def _call_cache_api(prompt: str) -> dict:
    """
    POSTs to the upstream Cache API at Port 8000 and returns the parsed JSON.

    Raises httpx.HTTPError on connection failures.
    The upstream response always has the shape:
        {
            "status":   "success",
            "source":   "RAM_Exact_Hit" | "DB_Semantic_Hit" | "LLM_Generation_Miss",
            "response": "<text>",
            "debug":    { "tier": ..., "cached": ..., ... }
        }
    """
    client = await get_http_client()
    upstream_response = await client.post(
        f"{CACHE_API_URL}/query",
        json={"prompt": prompt},
    )
    upstream_response.raise_for_status()
    return upstream_response.json()

# ---------------------------------------------------------------------------
# Helper — translate upstream debug dict → CacheDebugInfo
# ---------------------------------------------------------------------------

def _parse_debug(debug: dict) -> CacheDebugInfo:
    return CacheDebugInfo(
        tier=debug.get("tier"),
        similarity_score=debug.get("similarity_score"),
        reranker_score=debug.get("reranker_score"),
        hit_count=debug.get("hit_count"),
        classifier_called=debug.get("classifier_called"),
    )

# ---------------------------------------------------------------------------
# Helper — update session stats
# ---------------------------------------------------------------------------

def _record_request(hit: bool, latency_ms: float):
    _stats["requests_served"] += 1
    _stats["total_latency_ms"] += latency_ms
    if hit:
        _stats["cache_hits"] += 1
    else:
        _stats["cache_misses"] += 1

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + readiness probe",
    tags=["System"],
)
async def health_check():
    """
    Returns service status and whether the upstream cache (Port 8000) is reachable.
    Safe to call without an API key — intended for AWS ALB health checks.
    """
    upstream_status = "ok"
    try:
        client = await get_http_client()
        resp = await client.get(f"{CACHE_API_URL}/health", timeout=3.0)
        if resp.status_code != 200:
            upstream_status = f"degraded (HTTP {resp.status_code})"
    except Exception as exc:
        upstream_status = f"unreachable ({type(exc).__name__})"
        log.warning("[HEALTH] Upstream cache unreachable: %s", exc)

    return HealthResponse(
        status="ok",
        upstream_cache=upstream_status,
    )


@app.post(
    "/v1/cache/query",
    response_model=CacheQueryResponse,
    summary="Look up a query in the Semantic Cache",
    tags=["Cache"],
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {"description": "Cache HIT or MISS result"},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        502: {"model": ErrorResponse, "description": "Upstream cache service unavailable"},
    },
)
async def cache_query(body: CacheQueryRequest) -> CacheQueryResponse:
    """
    Main endpoint. Called by your application before any LLM call.

    **Flow:**
    - `cache_hit: true`  → use `response` directly, skip LLM call.
    - `cache_hit: false` → proceed to your LLM. `classification` tells you
      whether the query is GENERAL or PERSONAL (informational only).
    """
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    log.info(
        "[QUERY] request_id=%s prompt='%.60s...' user_id=%s",
        request_id, body.prompt, body.user_id or "—"
    )

    try:
        upstream = await _call_cache_api(body.prompt)
    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t_start) * 1000
        log.error("[QUERY] request_id=%s TIMEOUT calling upstream cache", request_id)
        _record_request(hit=False, latency_ms=latency_ms)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=ErrorResponse(
                request_id=request_id,
                error_code="UPSTREAM_TIMEOUT",
                message="The cache service timed out. Please retry.",
                latency_ms=round(latency_ms, 2),
            ).model_dump(),
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        log.error("[QUERY] request_id=%s upstream HTTP error: %s", request_id, exc)
        _record_request(hit=False, latency_ms=latency_ms)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=ErrorResponse(
                request_id=request_id,
                error_code="UPSTREAM_UNAVAILABLE",
                message="The cache service is temporarily unavailable.",
                latency_ms=round(latency_ms, 2),
            ).model_dump(),
        )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    source: str = upstream.get("source", "Cache_Miss")
    is_hit: bool = upstream.get("source") in {"RAM_Exact_Hit", "DB_Semantic_Hit"}
    cached_response: Optional[str] = upstream.get("response") if is_hit else None
    classification: Optional[str] = upstream.get("classification")
    debug_raw: dict = upstream.get("debug", {})

    _record_request(hit=is_hit, latency_ms=latency_ms)

    log.info(
        "[QUERY] request_id=%s cache_hit=%s source=%s latency=%.1fms",
        request_id, is_hit, source, latency_ms,
    )

    return CacheQueryResponse(
        request_id=request_id,
        cache_hit=is_hit,
        source=source,
        response=cached_response,
        classification=classification,
        latency_ms=latency_ms,
        debug=_parse_debug(debug_raw),
    )



@app.get(
    "/v1/stats",
    response_model=StatsResponse,
    summary="Session-level cache performance statistics",
    tags=["System"],
    dependencies=[Depends(verify_api_key)],
)
async def get_stats() -> StatsResponse:
    """
    Returns hit/miss counts and latency stats for the current server session.
    Resets when the bridge process restarts.
    """
    served = _stats["requests_served"]
    hits   = _stats["cache_hits"]
    misses = _stats["cache_misses"]
    total_lat = _stats["total_latency_ms"]

    hit_rate  = round((hits / served * 100) if served > 0 else 0.0, 2)
    avg_lat   = round((total_lat / served) if served > 0 else 0.0, 2)

    return StatsResponse(
        requests_served=served,
        cache_hits=hits,
        cache_misses=misses,
        hit_rate_pct=hit_rate,
        avg_latency_ms=avg_lat,
    )


# ---------------------------------------------------------------------------
# Global exception handler — always return a structured error envelope
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("[BRIDGE] Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred. Please try again or contact the service operator.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Dev entrypoint — `python axiom_bridge.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "axiom_bridge:app",
        host="0.0.0.0",
        port=BRIDGE_PORT,
        reload=False,
        log_level="info",
    )
