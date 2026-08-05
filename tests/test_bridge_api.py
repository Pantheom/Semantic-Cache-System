"""
tests/test_bridge_api.py — Integration tests for the Semantic Cache Bridge API.

These tests use FastAPI's TestClient (synchronous wrapper over httpx) and
mock the upstream Cache API (Port 8000) so no real services need to be running.

Run:
    pip install pytest pytest-asyncio
    pytest tests/test_bridge_api.py -v
"""

from __future__ import annotations

from typing import Optional
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Set env vars BEFORE importing axiom_bridge so the startup validation passes.
os.environ.setdefault("BRIDGE_API_KEY", "test-secret-key-do-not-use-in-prod")
os.environ.setdefault("CACHE_API_URL",  "http://127.0.0.1:8000")
os.environ.setdefault("BRIDGE_PORT",    "8002")

from fastapi.testclient import TestClient
from axiom_bridge import app  # noqa: E402 — must come after env setup

client = TestClient(app)

VALID_KEY   = "test-secret-key-do-not-use-in-prod"
INVALID_KEY = "wrong-key"
AUTH_HEADER = {"X-API-Key": VALID_KEY}

# ---------------------------------------------------------------------------
# Fixtures — mock upstream responses
# ---------------------------------------------------------------------------

RAM_HIT_RESPONSE = {
    "status": "success",
    "source": "RAM_Exact_Hit",
    "response": "Machine learning is a subset of artificial intelligence...",
    "classification": "GENERAL",
    "debug": {
        "tier": "RAM",
        "cached": True,
        "classifier_called": True,
        "decision_layer": "HEURISTIC",
        "heuristic_reason": None,
    },
}

DB_HIT_RESPONSE = {
    "status": "success",
    "source": "DB_Semantic_Hit",
    "response": "Machine learning is a branch of AI...",
    "classification": "GENERAL",
    "debug": {
        "tier": "DB",
        "cached": True,
        "similarity_score": 0.9124,
        "reranker_score": 0.81,
        "hit_count": 7,
        "classifier_called": False,
    },
}

MISS_RESPONSE = {
    "status": "success",
    "source": "Cache_Miss",
    "response": None,
    "classification": "GENERAL",
    "debug": {
        "tier": "MISS",
        "cached": False,
        "classifier_called": True,
        "decision_layer": "NLI",
        "heuristic_reason": None,
    },
}

MISS_PERSONAL_RESPONSE = {
    "status": "success",
    "source": "Cache_Miss",
    "response": None,
    "classification": "PERSONAL",
    "debug": {
        "tier": "MISS",
        "cached": False,
        "classifier_called": True,
        "decision_layer": "HEURISTIC",
        "heuristic_reason": "Possessive + personal noun: 'my account'",
    },
}

STORE_SUCCESS_RESPONSE = {
    "status": "success",
    "classification": "GENERAL",
    "db_stored": True,
    "ram_stored": True,
    "decision_layer": "NLI",
    "heuristic_reason": None,
}


def _mock_upstream(upstream_json: dict, store_json: Optional[dict] = None):
    """Returns an async mock that simulates a successful upstream response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = upstream_json
    mock_response.raise_for_status = MagicMock()

    store_mock_response = MagicMock()
    store_mock_response.status_code = 200
    store_mock_response.json.return_value = store_json or STORE_SUCCESS_RESPONSE
    store_mock_response.raise_for_status = MagicMock()

    # post returns query response for /query, store response for /store
    async def mock_post(url, **kwargs):
        if "/store" in url:
            return store_mock_response
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.get  = AsyncMock(return_value=MagicMock(status_code=200))
    mock_client.is_closed = False
    return mock_client


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        """GET /health should return 200 and upstream_cache: ok (no auth needed)."""
        mock_client = _mock_upstream({})
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "axiom_bridge"
        assert "upstream_cache" in data

    def test_health_no_auth_required(self):
        """Health endpoint must be reachable WITHOUT an API key."""
        mock_client = _mock_upstream({})
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.get("/health")  # no X-API-Key header
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_api_key_returns_401(self):
        """Requests without X-API-Key header must be rejected with 401."""
        resp = client.post(
            "/v1/cache/query",
            json={"prompt": "What is machine learning?"},
            # No X-API-Key header
        )
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self):
        """Requests with an incorrect X-API-Key must be rejected with 401."""
        resp = client.post(
            "/v1/cache/query",
            json={"prompt": "What is machine learning?"},
            headers={"X-API-Key": INVALID_KEY},
        )
        assert resp.status_code == 401

    def test_correct_api_key_passes(self):
        """Requests with the correct X-API-Key should not be rejected for auth."""
        mock_client = _mock_upstream(RAM_HIT_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.post(
                "/v1/cache/query",
                json={"prompt": "What is machine learning?"},
                headers=AUTH_HEADER,
            )
        # Auth passed — any non-401 status is acceptable here
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# POST /v1/cache/query — cache hits
# ---------------------------------------------------------------------------

class TestCacheQuery:
    def test_ram_hit_returns_cache_hit_true(self):
        """A RAM cache hit should return cache_hit=True with classification."""
        mock_client = _mock_upstream(RAM_HIT_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.post(
                "/v1/cache/query",
                json={"prompt": "What is machine learning?"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is True
        assert data["source"] == "RAM_Exact_Hit"
        assert data["response"] is not None
        assert data["classification"] == "GENERAL"
        assert "request_id" in data
        assert "latency_ms" in data

    def test_db_hit_returns_cache_hit_true_with_scores(self):
        """A DB semantic hit should return cache_hit=True, classification=GENERAL, and scores."""
        mock_client = _mock_upstream(DB_HIT_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.post(
                "/v1/cache/query",
                json={"prompt": "Tell me about machine learning"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is True
        assert data["source"] == "DB_Semantic_Hit"
        assert data["classification"] == "GENERAL"   # always GENERAL for DB entries
        assert data["debug"]["similarity_score"] == pytest.approx(0.9124, rel=1e-3)
        assert data["debug"]["reranker_score"] == pytest.approx(0.81, rel=1e-3)
        assert data["debug"]["hit_count"] == 7

    def test_cache_miss_returns_cache_hit_false_null_response(self):
        """
        A cache miss must return cache_hit=False, response=None, and a classification.
        GENERAL miss → your application should call its LLM and optionally cache the result.
        PERSONAL miss → your application must NOT cache this result (privacy).
        """
        mock_client = _mock_upstream(MISS_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.post(
                "/v1/cache/query",
                json={"prompt": "Explain quantum entanglement in simple terms"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is False
        assert data["source"] == "Cache_Miss"
        assert data["response"] is None
        assert data["classification"] in {"GENERAL", "PERSONAL"}

    def test_personal_miss_classification(self):
        """A personal query miss should return classification=PERSONAL."""
        mock_client = _mock_upstream(MISS_PERSONAL_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.post(
                "/v1/cache/query",
                json={"prompt": "What is my account balance?"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is False
        assert data["classification"] == "PERSONAL"  # your application must NOT cache this result (privacy)

    def test_optional_fields_accepted(self):
        """user_id and session_id are optional — should not cause validation errors."""
        mock_client = _mock_upstream(RAM_HIT_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            resp = client.post(
                "/v1/cache/query",
                json={
                    "prompt": "What is Python?",
                    "user_id": "user-abc-123",
                    "session_id": "sess-xyz-789",
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    def test_empty_prompt_rejected(self):
        """Empty prompt should be rejected with 422 Unprocessable Entity."""
        resp = client.post(
            "/v1/cache/query",
            json={"prompt": ""},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422

    def test_response_includes_unique_request_id(self):
        """Each response must have a unique request_id (UUID)."""
        mock_client = _mock_upstream(RAM_HIT_RESPONSE)
        with patch("axiom_bridge._http_client", mock_client):
            r1 = client.post("/v1/cache/query", json={"prompt": "foo"}, headers=AUTH_HEADER)
            r2 = client.post("/v1/cache/query", json={"prompt": "bar"}, headers=AUTH_HEADER)
        assert r1.json()["request_id"] != r2.json()["request_id"]



# ---------------------------------------------------------------------------
# GET /v1/stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_expected_fields(self):
        """Stats endpoint should return hit/miss counts and latency."""
        resp = client.get("/v1/stats", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "requests_served" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "hit_rate_pct" in data
        assert "avg_latency_ms" in data

    def test_stats_requires_auth(self):
        """Stats endpoint must require X-API-Key."""
        resp = client.get("/v1/stats")
        assert resp.status_code == 401

