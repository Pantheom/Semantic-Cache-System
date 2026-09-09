import os
import requests
import math
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="Hybrid Semantic Cache API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config & Initialization ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-anon-key")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Device Detection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"[DEVICE] GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) — Loading model on GPU")
else:
    print("[DEVICE] No GPU detected — Loading model on CPU")

print(f"Loading Embedding Model on {device.upper()}...")
embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)

print("Loading Cross-Encoder Reranker on CPU (saving GPU VRAM)...")
reranker_model = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

# In-Memory Exact Match Cache
# Key: Query String -> Value: Response String
ram_cache = {}
RAM_LIMIT            = 10000
SIMILARITY_THRESHOLD = 0.85
RERANKER_THRESHOLD   = 0.70

# Classifier service — runs separately on port 8001
CLASSIFIER_URL = "http://127.0.0.1:8001/classify"
CLASSIFIER_TIMEOUT = 5   # seconds

class QueryRequest(BaseModel):
    prompt: str

class StoreRequest(BaseModel):
    prompt:   str = Field(..., min_length=1, description="The user prompt to store.")
    response: str = Field(..., min_length=1, description="The LLM response to cache.")

# --- Helpers ---

def update_ram_cache(query: str, response: str):
    if len(ram_cache) >= RAM_LIMIT:
        oldest_key = next(iter(ram_cache))
        del ram_cache[oldest_key]
    ram_cache[query] = response

def record_hit(row_id: str):
    """
    Background task — atomically increments hit_count and refreshes
    last_accessed_at for the matched DB row via the increment_hit() RPC.

    row_id is a UUID string (Supabase returns uuid columns as str in Python).
    Called via FastAPI BackgroundTasks so it never blocks the response.
    A failure here is logged but does not affect the user-facing result.
    """
    try:
        supabase.rpc("increment_hit", {"p_row_id": str(row_id)}).execute()
        print(f"[HIT TRACK] Row {row_id} hit_count incremented.")
    except Exception as e:
        print(f"[HIT TRACK] Failed to increment hit for row {row_id}: {e}")

def classify_query(query: str) -> dict:
    """
    Calls the classifier service (port 8001) to decide PERSONAL vs GENERAL.

    Called on every request — on hits to inform the calling backend of query type,
    on misses so the calling backend knows whether this query is safe to cache
    (GENERAL = cacheable, PERSONAL = skip storage).

    Defaults to PERSONAL if the classifier is unreachable:
    better to skip caching than to risk storing private data.

    Returns a dict:
      { "label": "GENERAL" | "PERSONAL",
        "decision_layer": "HEURISTIC" | "NLI" | "UNAVAILABLE",
        "heuristic_reason": str | None }
    """
    try:
        resp   = requests.post(CLASSIFIER_URL, json={"query": query}, timeout=CLASSIFIER_TIMEOUT)
        result = resp.json()
        label  = result.get("final_label", "PERSONAL")
        layer  = result.get("decision_layer", "?")
        reason = result.get("heuristic_reason")
        print(f"[CLASSIFIER] '{query[:45]}' → {label} (layer: {layer}" +
              (f", reason: {reason}" if reason else "") + ")")
        return {"label": label, "decision_layer": layer, "heuristic_reason": reason}
    except Exception as e:
        print(f"[CLASSIFIER] Unreachable: {e} — defaulting to PERSONAL")
        return {"label": "PERSONAL", "decision_layer": "UNAVAILABLE", "heuristic_reason": None}

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "cache_api", "device": device}


# --- Cache Store Endpoint ---
@app.post("/store")
async def store_entry(request: StoreRequest):
    """
    Compute embedding for the prompt and insert into shared_llm_cache.
    No duplicate check — the caller (bridge) is responsible for only
    calling this on a confirmed cache miss.
    """
    prompt_text   = request.prompt.strip()
    response_text = request.response.strip()

    # Compute embedding using the already-loaded model
    embedding = embedding_model.encode(prompt_text).tolist()

    result = supabase.table("shared_llm_cache").insert({
        "query_text":    prompt_text,
        "response_text": response_text,
        "embedding":     embedding,
    }).execute()

    row_id = result.data[0]["id"] if result.data else None
    print(f"[STORE] Inserted row id={row_id} query='{prompt_text[:50]}'")
    return {"status": "ok", "id": row_id}

# --- Core Query Endpoint ---
@app.post("/query")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Three-tier cache lookup. Returns a hit (with cached response + classification)
    or a miss (with classification). No LLM generation happens here — that is
    handled by your application.

    Classification is returned on every response:
      GENERAL  → query is about general knowledge — safe to cache
      PERSONAL → query is about user-specific data — do not cache
    """
    user_prompt = request.prompt.strip()

    # 1. Tier 1: RAM Exact Match
    if user_prompt in ram_cache:
        # Run classifier so the calling backend knows the query type even on hits.
        # Heuristic layer is <1ms so latency impact on RAM hits is negligible.
        classification = classify_query(user_prompt)
        print(f"[RAM HIT]  '{user_prompt[:50]}'")
        return {
            "status":          "success",
            "source":          "RAM_Exact_Hit",
            "response":        ram_cache[user_prompt],
            "classification":  classification["label"],
            "debug": {
                "tier":               "RAM",
                "cached":             True,
                "classifier_called":  True,
                "decision_layer":     classification["decision_layer"],
                "heuristic_reason":   classification["heuristic_reason"],
            }
        }

    # 2. Embed the query
    query_vector = embedding_model.encode(user_prompt).tolist()

    # 3. Tier 2: Semantic DB Match
    try:
        db_search = supabase.rpc(
            "match_shared_cache",
            {
                "query_embedding": query_vector,
                "match_threshold": SIMILARITY_THRESHOLD,
                "match_count":     1
            }
        ).execute()

        if db_search.data:
            matched_row     = db_search.data[0]
            matched_id      = matched_row["id"]
            matched_query   = matched_row["query_text"]
            cached_response = matched_row["response_text"]
            similarity      = round(matched_row["similarity"], 4)
            current_hits    = matched_row.get("hit_count", 0)

            # Tier 2b: Cross-Encoder Reranker Verification
            raw_logit      = reranker_model.predict([user_prompt, matched_query])
            reranker_score = round(1 / (1 + math.exp(-float(raw_logit))), 4)

            if reranker_score >= RERANKER_THRESHOLD:
                # Backfill RAM for next time
                update_ram_cache(user_prompt, cached_response)
                print(f"[DB HIT]   '{user_prompt[:50]}' (similarity: {similarity}, reranker: {reranker_score}, hits: {current_hits})")

                # Non-blocking hit tracking — fires after response is sent.
                background_tasks.add_task(record_hit, matched_id)

                # DB entries are always GENERAL by definition — personal queries
                # are never written to the shared DB (Privacy Gatekeeper rule).
                return {
                    "status":          "success",
                    "source":          "DB_Semantic_Hit",
                    "response":        cached_response,
                    "classification":  "GENERAL",
                    "debug": {
                        "tier":               "DB",
                        "cached":             True,
                        "similarity_score":   similarity,
                        "reranker_score":     reranker_score,
                        "reranker_model":     "BAAI/bge-reranker-base",
                        "hit_count":          current_hits + 1,
                        "classifier_called":  False,
                        "classifier_note":    "DB entries are always GENERAL — no classifier needed",
                    }
                }
            else:
                print(f"[CACHE REJECT] '{user_prompt[:45]}' matched DB '{matched_query[:45]}' "
                      f"(similarity: {similarity}) but rejected by reranker ({reranker_score} < {RERANKER_THRESHOLD})")
    except Exception as e:
        print(f"Vector/Reranker search failed: {e}")

    # 4. Cache Miss — no LLM generation here.
    # Classify so the calling application knows whether this query type is
    # safe to cache after LLM generation:
    #   GENERAL  → safe to cache (general world knowledge)
    #   PERSONAL → do not cache (user-specific data, privacy protection)
    classification = classify_query(user_prompt)
    query_type     = classification["label"]
    print(f"[MISS]     '{user_prompt[:50]}' — classification: {query_type}")

    return {
        "status":          "success",
        "source":          "Cache_Miss",
        "response":        None,
        "classification":  query_type,
        "debug": {
            "tier":               "MISS",
            "cached":             False,
            "classifier_called":  True,
            "decision_layer":     classification["decision_layer"],
            "heuristic_reason":   classification["heuristic_reason"],
        }
    }