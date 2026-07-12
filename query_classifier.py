"""
AXIOM Semantic Cache — Query Classifier Service
===============================================
Classifies incoming queries as PERSONAL or GENERAL before they reach the cache.

PERSONAL  → bypass cache entirely (response may contain private user data)
GENERAL   → safe to look up / store in the shared semantic cache

Two classification layers:
  Layer 1 — Heuristic  (0ms)   : regex + personal noun matching
  Layer 2 — NLI        (~10ms) : zero-shot cross-encoder (deberta-v3-small)

Run on a separate port so it doesn't conflict with main.py:
    uvicorn query_classifier:app --port 8001 --reload

Endpoints:
  POST /classify             → combined result (heuristic → NLI fallback)
  POST /classify/heuristic   → heuristic layer only
  POST /classify/nli         → NLI layer only
  POST /classify/compare     → both layers side-by-side (for testing/debugging)
  GET  /health               → service health check
"""

import re
import time
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AXIOM Query Classifier",
    description="Classifies queries as PERSONAL or GENERAL before semantic cache lookup.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# NLI Model — loaded once at startup
# ---------------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1   # 0 = first GPU, -1 = CPU
device_label = "GPU" if device == 0 else "CPU"
print(f"[CLASSIFIER] Loading NLI model on {device_label}...")

nli_classifier = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=device,
)
print("[CLASSIFIER] NLI model ready.")

# Hypothesis used for zero-shot classification.
# If the query ENTAILS this hypothesis → it's personal.
NLI_HYPOTHESIS = (
    "This question requires knowing who the specific user is, "
    "or accessing their private account data, to give a correct answer."
)

# Threshold: if P(personal) exceeds this → classify as PERSONAL
# Lower = more aggressive filtering (fewer false negatives, more false positives)
NLI_PERSONAL_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Heuristic Classifier
# ---------------------------------------------------------------------------

# Nouns that, when paired with "my/mine/our", signal user-specific data
PERSONAL_NOUNS = {
    # Financial
    "account", "balance", "transaction", "transactions", "invoice", "bill",
    "bills", "salary", "payment", "payments", "subscription", "card",
    "statement", "loan", "mortgage", "tax", "taxes",
    "refund", "expense", "expenses",
    # Identity / Contact
    "name", "email", "address", "phone", "number", "password", "pin",
    "profile", "id", "ssn", "identity", "username", "login",
    # Schedules / Events
    "order", "orders", "booking", "appointment", "appointments",
    "schedule", "calendar", "reservation", "reservations", "delivery",
    # Personal files / data
    "file", "files", "photo", "photos", "document", "documents",
    "data", "record", "records", "history", "report", "reports",
    "prescription", "medical", "insurance",
    # Relationships / comms
    "message", "messages", "chat", "notification", "inbox",
    "contact", "contacts",
}

# Action phrases that inherently require user identity, regardless of nouns
PERSONAL_ACTION_PATTERNS = [
    r"\bremind me\b",
    r"\bbook (?:it |this |a |an )?for me\b",
    r"\bcancel my\b",
    r"\bshow me my\b",
    r"\bwhat did i\b",
    r"\bmy last\b",
    r"\blog me (in|out)\b",
    r"\breset my\b",
    r"\bsend (?:it |this )?on my behalf\b",
    r"\btrack my\b",
    r"\bwhere is my\b",
    r"\bwhen (is|was|will) my\b",
    r"\bhow much (do i|did i|have i)\b",
    r"\bwhat (is|are|was|were) my\b",
    r"\bupdate my\b",
    r"\bchange my\b",
    r"\bdelete my\b",
    # Self-description patterns — NLI misses these
    r"\btell me about my\b",        # "tell me about my life / past / job"
    r"\bdescribe my\b",             # "describe my situation"
    r"\bmy life\b",                 # "tell me about my life", "my life story"
    r"\bmy story\b",
    r"\bmy situation\b",
    r"\bmy background\b",
    r"\bmy experience\b",
    r"\bmy journey\b",
]

# Compile patterns once at startup
_compiled_actions = [re.compile(p, re.IGNORECASE) for p in PERSONAL_ACTION_PATTERNS]

# "my/mine/our" followed (within 3 words) by a personal noun
_MY_NOUN_RE = re.compile(
    r"\b(my|mine|our)\b\s+(?:\w+\s+){0,2}(" + "|".join(PERSONAL_NOUNS) + r")\b",
    re.IGNORECASE,
)

# PII patterns — direct red flags regardless of context
_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),                     # SSN
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),            # credit card
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),   # email
    re.compile(r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
]


def heuristic_classify(query: str) -> dict:
    """
    Fast rule-based classifier.
    Returns dict with keys: label, reason, latency_ms
    """
    t0 = time.perf_counter()
    q  = query.strip()

    # 1. PII pattern check
    for pattern in _PII_PATTERNS:
        if pattern.search(q):
            return {
                "label": "PERSONAL",
                "reason": "PII pattern detected (email / phone / card / SSN)",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    # 2. Personal action phrases
    for pattern in _compiled_actions:
        m = pattern.search(q)
        if m:
            return {
                "label": "PERSONAL",
                "reason": f"Personal action phrase matched: '{m.group()}'",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    # 3. "my / our" + personal noun
    m = _MY_NOUN_RE.search(q)
    if m:
        return {
            "label": "PERSONAL",
            "reason": f"Possessive + personal noun: '{m.group()}'",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    return {
        "label": "GENERAL",
        "reason": None,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# NLI Classifier
# ---------------------------------------------------------------------------

def nli_classify(query: str) -> dict:
    """
    Zero-shot NLI classifier.
    Uses descriptive candidate labels so the model can reason about
    what "personal" and "general" actually mean — not just the words.
    Returns dict with keys: label, confidence, scores, latency_ms
    """
    t0 = time.perf_counter()

    result = nli_classifier(
        query,
        # Shorter, direct labels — avoids the model pattern-matching on long
        # descriptive phrases and triggering on first-person phrasing like "i want".
        candidate_labels=[
            "about the speaker's own private account data",
            "about general world knowledge",
        ],
        hypothesis_template="This question is {}.",
    )

    scores = dict(zip(result["labels"], result["scores"]))
    personal_score = scores.get("about the speaker's own private account data", 0.0)
    general_score  = scores.get("about general world knowledge",                 0.0)

    label = "PERSONAL" if personal_score >= NLI_PERSONAL_THRESHOLD else "GENERAL"

    return {
        "label":      label,
        "confidence": round(personal_score if label == "PERSONAL" else general_score, 4),
        "scores": {
            "PERSONAL": round(personal_score, 4),
            "GENERAL":  round(general_score,  4),
        },
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

class HeuristicResponse(BaseModel):
    query:      str
    label:      str
    reason:     str | None
    latency_ms: float

class NLIResponse(BaseModel):
    query:      str
    label:      str
    confidence: float
    scores:     dict
    latency_ms: float

class CombinedResponse(BaseModel):
    query:            str
    final_label:      str
    decision_layer:   str           # "HEURISTIC" | "NLI"
    heuristic_label:  str
    heuristic_reason: str | None
    nli_label:        str | None    # None if heuristic was definitive
    nli_confidence:   float | None
    nli_scores:       dict | None
    total_latency_ms: float

class CompareResponse(BaseModel):
    query:     str
    heuristic: dict
    nli:       dict
    agreement: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "nli_model":     "cross-encoder/nli-deberta-v3-small",
        "device":        device_label,
        "nli_threshold": NLI_PERSONAL_THRESHOLD,
    }


@app.post("/classify/heuristic", response_model=HeuristicResponse)
def classify_heuristic(req: QueryRequest):
    """Heuristic layer only — instant rule-based check."""
    result = heuristic_classify(req.query)
    return HeuristicResponse(query=req.query, **result)


@app.post("/classify/nli", response_model=NLIResponse)
def classify_nli(req: QueryRequest):
    """NLI layer only — zero-shot cross-encoder inference."""
    result = nli_classify(req.query)
    return NLIResponse(query=req.query, **result)


@app.post("/classify", response_model=CombinedResponse)
def classify_combined(req: QueryRequest):
    """
    Combined classifier (recommended for production):
      1. Run heuristic (0ms) — if PERSONAL, return immediately
      2. If GENERAL, confirm with NLI — NLI result is final answer
    """
    t_start = time.perf_counter()

    h = heuristic_classify(req.query)

    if h["label"] == "PERSONAL":
        return CombinedResponse(
            query            = req.query,
            final_label      = "PERSONAL",
            decision_layer   = "HEURISTIC",
            heuristic_label  = "PERSONAL",
            heuristic_reason = h["reason"],
            nli_label        = None,
            nli_confidence   = None,
            nli_scores       = None,
            total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2),
        )

    # Heuristic passed — run NLI as second opinion
    n = nli_classify(req.query)

    return CombinedResponse(
        query            = req.query,
        final_label      = n["label"],
        decision_layer   = "NLI",
        heuristic_label  = "GENERAL",
        heuristic_reason = None,
        nli_label        = n["label"],
        nli_confidence   = n["confidence"],
        nli_scores       = n["scores"],
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2),
    )


@app.post("/classify/compare", response_model=CompareResponse)
def classify_compare(req: QueryRequest):
    """
    Runs both layers independently and returns results side-by-side.
    Use this endpoint to test and calibrate the two approaches against each other.
    """
    h = heuristic_classify(req.query)
    n = nli_classify(req.query)

    return CompareResponse(
        query     = req.query,
        heuristic = h,
        nli       = n,
        agreement = h["label"] == n["label"],
    )
