import os
import requests
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# In-Memory Exact Match Cache
# Key: Query String -> Value: Response String
ram_cache = {}
RAM_LIMIT            = 10000
SIMILARITY_THRESHOLD = 0.85

# Classifier service — runs separately on port 8001
CLASSIFIER_URL = "http://127.0.0.1:8001/classify"
CLASSIFIER_TIMEOUT = 5   # seconds

class QueryRequest(BaseModel):
    prompt: str

# --- Helpers ---
def generate_llm_response(prompt: str) -> str:
    """
    Placeholder for your actual LLM call (e.g., Groq, OpenAI).
    """
    return f"[Live Generated] I am a newly generated response for: {prompt}"

def update_ram_cache(query: str, response: str):
    if len(ram_cache) >= RAM_LIMIT:
        oldest_key = next(iter(ram_cache))
        del ram_cache[oldest_key]
    ram_cache[query] = response

def classify_query(query: str) -> dict:
    """
    Calls the classifier service (port 8001) to decide PERSONAL vs GENERAL.
    Called only on cache misses — zero overhead on cache hits.

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
        print(f"[CLASSIFIER] Unreachable: {e} — defaulting to PERSONAL (DB skipped)")
        return {"label": "PERSONAL", "decision_layer": "UNAVAILABLE", "heuristic_reason": None}

# --- Core Routing Engine ---
@app.post("/query")
async def process_query(request: QueryRequest):
    user_prompt = request.prompt.strip()

    # 1. Tier 1: Lexical RAM Match
    if user_prompt in ram_cache:
        print(f"[RAM HIT]  '{user_prompt[:50]}'")
        return {
            "status":   "success",
            "source":   "RAM_Exact_Hit",
            "response": ram_cache[user_prompt],
            "debug": {
                "tier":               "RAM",
                "cached":             True,
                "classifier_called":  False,
                "classifier_note":    "Classifier not called on cache hits",
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
                "match_count": 1
            }
        ).execute()
        
        if db_search.data:
            matched_row    = db_search.data[0]
            cached_response = matched_row["response_text"]
            similarity      = round(matched_row["similarity"], 4)

            # Backfill RAM for next time
            update_ram_cache(user_prompt, cached_response)
            print(f"[DB HIT]   '{user_prompt[:50]}' (similarity: {similarity})")

            return {
                "status":   "success",
                "source":   "DB_Semantic_Hit",
                "response": cached_response,
                "debug": {
                    "tier":               "DB",
                    "cached":             True,
                    "similarity_score":   similarity,
                    "classifier_called":  False,
                    "classifier_note":    "Classifier not called on cache hits",
                }
            }
    except Exception as e:
        print(f"Vector search failed: {e}")

    # 4. Cache Miss: Generate fresh response
    fresh_response = generate_llm_response(user_prompt)

    # Always store in RAM — exact-match only, ephemeral, not semantically searchable
    update_ram_cache(user_prompt, fresh_response)

    # --- Gatekeeper: only persist to shared DB if query is GENERAL ---
    # Classification happens here (at storage time) — zero cost on cache hits.
    classification = classify_query(user_prompt)
    query_type     = classification["label"]
    db_stored      = False

    if query_type == "GENERAL":
        try:
            supabase.table("shared_llm_cache").insert({
                "query_text":    user_prompt,
                "response_text": fresh_response,
                "embedding":     query_vector
            }).execute()
            db_stored = True
            print(f"[DB STORE] General query stored in DB.")
        except Exception as e:
            print(f"[DB STORE] Insert failed: {e}")
    else:
        print(f"[DB SKIP]  Personal query — DB storage skipped.")

    return {
        "status":   "success",
        "source":   "LLM_Generation_Miss",
        "response": fresh_response,
        "debug": {
            "tier":               "LLM",
            "cached":             False,
            "classifier_called":  True,
            "query_type":         query_type,
            "decision_layer":     classification["decision_layer"],
            "heuristic_reason":   classification["heuristic_reason"],
            "db_stored":          db_stored,
        }
    }