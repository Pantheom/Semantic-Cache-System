"""
bridge_models.py — Pydantic schemas for the Semantic Cache Bridge API.

All request and response models are defined here and imported by axiom_bridge.py.
Keeping them separate keeps axiom_bridge.py clean and makes schemas easy to evolve.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CacheQueryRequest(BaseModel):
    """
    Sent by the calling application to look up a query in the cache.

    Fields
    ------
    prompt      : The user's query text (required).
    user_id     : Optional caller-supplied user identifier.
                  Passed through for logging only — the bridge does NOT use it
                  to scope cache lookups (the Cache Privacy Classifier handles
                  personal-query detection internally).
    session_id  : Optional session identifier for traceability.
    """
    prompt: str = Field(..., min_length=1, max_length=4096,
                        description="The query to look up in the semantic cache.")
    user_id: Optional[str] = Field(None, description="Optional user UUID for logging.")
    session_id: Optional[str] = Field(None, description="Optional session ID for traceability.")

    model_config = {"json_schema_extra": {
        "example": {
            "prompt": "What is machine learning?",
            "user_id": "a1b2c3d4-0000-0000-0000-000000000001",
            "session_id": "sess-xyz-001"
        }
    }}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CacheDebugInfo(BaseModel):
    """
    Supplementary debug metadata included in every cache lookup response.
    All fields are nullable — they are only populated when relevant.
    """
    tier: Optional[str] = Field(
        None,
        description="Which cache tier answered: 'RAM', 'DB', or 'LLM' (miss)."
    )
    similarity_score: Optional[float] = Field(
        None,
        description="Cosine similarity score from the vector DB (DB hits only)."
    )
    reranker_score: Optional[float] = Field(
        None,
        description="Cross-encoder reranker confidence score (DB hits only)."
    )
    hit_count: Optional[int] = Field(
        None,
        description="Lifetime hit count for this cache entry (DB hits only)."
    )
    classifier_called: Optional[bool] = Field(
        None,
        description="Whether the Cache Privacy Classifier ran (misses only)."
    )


class CacheQueryResponse(BaseModel):
    """
    Returned to the calling application after a cache lookup.

    cache_hit == True  → use `response` directly, skip Model Router.
    cache_hit == False → proceed to Model Router. `classification` tells
                         you whether the query is GENERAL or PERSONAL
                         (informational — storage is handled externally).

    Fields
    ------
    request_id      : UUID generated per request for end-to-end traceability.
    cache_hit       : True if a cached response was found and returned.
    source          : Which tier answered: RAM_Exact_Hit | DB_Semantic_Hit | Cache_Miss
    response        : The cached answer text, or null on a miss.
    classification  : GENERAL or PERSONAL — always populated.
                      Informational on hits; actionable on misses (tells
                      calling application whether this query type is cacheable).
    latency_ms      : Total round-trip time inside the bridge.
    debug           : Supplementary metadata (similarity scores, tier info).
    """
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Per-request UUID for traceability."
    )
    cache_hit: bool = Field(..., description="True if a cached response was returned.")
    source: str = Field(
        ...,
        description="One of: RAM_Exact_Hit | DB_Semantic_Hit | Cache_Miss"
    )
    response: Optional[str] = Field(
        None,
        description="The cached response text. Null on a cache miss."
    )
    classification: Optional[str] = Field(
        None,
        description="GENERAL or PERSONAL. DB hits are always GENERAL by definition."
    )
    latency_ms: float = Field(..., description="Bridge round-trip latency in milliseconds.")
    debug: CacheDebugInfo = Field(default_factory=CacheDebugInfo)

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "summary": "Cache HIT (RAM)",
                "value": {
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "cache_hit": True,
                    "source": "RAM_Exact_Hit",
                    "response": "Machine learning is a subset of AI...",
                    "latency_ms": 8.2,
                    "debug": {
                        "tier": "RAM",
                        "similarity_score": None,
                        "reranker_score": None,
                        "hit_count": None,
                        "classifier_called": False
                    }
                }
            },
            {
                "summary": "Cache MISS → your application should call its LLM",
                "value": {
                    "request_id": "660e8400-e29b-41d4-a716-446655440001",
                    "cache_hit": False,
                    "source": "LLM_Generation_Miss",
                    "response": None,
                    "latency_ms": 78.5,
                    "debug": {
                        "tier": "LLM",
                        "similarity_score": None,
                        "reranker_score": None,
                        "hit_count": None,
                        "classifier_called": True
                    }
                }
            }
        ]
    }}


class HealthResponse(BaseModel):
    """Liveness + readiness response for AWS health checks."""
    status: str = Field(..., description="'ok' when the service is healthy.")
    service: str = "axiom_bridge"
    version: str = "1.0.0"
    upstream_cache: str = Field(
        ...,
        description="'ok' if Port 8000 is reachable, 'unreachable' otherwise."
    )


class StatsResponse(BaseModel):
    """Session-level statistics returned by GET /v1/stats."""
    service: str = "axiom_bridge"
    requests_served: int = Field(..., description="Total requests handled this session.")
    cache_hits: int = Field(..., description="Requests that returned a cached response.")
    cache_misses: int = Field(..., description="Requests that resulted in a cache miss.")
    hit_rate_pct: float = Field(..., description="Hit rate as a percentage (0–100).")
    avg_latency_ms: float = Field(..., description="Average bridge latency this session.")


class ErrorResponse(BaseModel):
    """Standard error envelope returned on all failure paths."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "error"
    error_code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    latency_ms: Optional[float] = None
