import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED_COUNT        = 2000          # Top-up run — 2000 more rows
BATCH_SIZE        = 50            # Embed in chunks of 50 on GPU, then insert row-by-row
MAX_QUERY_CHARS   = 500           # Skip queries longer than this
DATASET_NAME      = "sharegpt"
DATASET_FILE      = "./sharegpt_data/sharegpt_sample.jsonl"   # Local JSONL file
CHECKPOINT_PATH   = Path("./checkpoints/seed_checkpoint.json")

# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"[DEVICE] GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) — Loading model on GPU")
else:
    print("[DEVICE] No GPU detected — Loading model on CPU")

# ---------------------------------------------------------------------------
# Supabase Init
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-service-key")
print(f"[DEBUG] Initializing Supabase client with URL: {SUPABASE_URL}")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
print(f"[DEBUG] Loading embedding model ('all-mpnet-base-v2') on {device.upper()}...")
model = SentenceTransformer('all-mpnet-base-v2', device=device)

# ---------------------------------------------------------------------------
# Checkpoint — decide run mode
# ---------------------------------------------------------------------------
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

existing_queries: set[str] = set()   # used only on first-run (no checkpoint)
row_offset: int = 0                  # dataset rows to skip (checkpoint mode)

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "r") as f:
        checkpoint = json.load(f)

    row_offset       = checkpoint.get("last_inspected", 0)
    cumulative_total = checkpoint.get("last_inserted", 0)

    print(f"[CHECKPOINT] Found checkpoint — resuming after row {row_offset} "
          f"(previously inserted: {cumulative_total})")

else:
    print("[CHECKPOINT] No checkpoint — fetching existing queries from Supabase for dedup...")

    PAGE_SIZE = 1000
    page      = 0
    while True:
        response = (
            supabase.table("shared_llm_cache")
            .select("query_text")
            .range(page * PAGE_SIZE, (page + 1) * PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data
        if not rows:
            break
        for r in rows:
            existing_queries.add(r["query_text"])
        print(f"[DEDUP] Page {page + 1}: {len(existing_queries)} unique queries loaded...")
        if len(rows) < PAGE_SIZE:
            break
        page += 1

    cumulative_total = len(existing_queries)
    print(f"[DEDUP] Done — {len(existing_queries)} existing queries loaded.")

# ---------------------------------------------------------------------------
# Dataset — read from local JSONL file (one JSON object per line)
# JSONL is naturally memory-safe: each line is parsed individually
# ---------------------------------------------------------------------------
print(f"\n[DEBUG] Reading local file '{DATASET_FILE}'...")

# ---------------------------------------------------------------------------
# Main seeding loop — ijson streams items one-by-one, never loads full file
# ---------------------------------------------------------------------------
count           = 0    # new inserts this run
total_inspected = 0    # rows seen this run (for checkpoint offset)

print(f"[DEBUG] Target: {SEED_COUNT} inserts | Batch size: {BATCH_SIZE} | Offset: {row_offset}\n")

# Batch accumulators
batch_queries   = []   # instruction strings
batch_responses = []   # response strings

def flush_batch(batch_queries, batch_responses, count):
    """Embed in small chunks on GPU and insert strictly row-by-row with live debug logs."""
    if not batch_queries:
        return count

    vectors = model.encode(
        batch_queries,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    for j in range(len(batch_queries)):
        query_text = batch_queries[j]
        rec = {
            "query_text":    query_text,
            "response_text": batch_responses[j],
            "embedding":     vectors[j].tolist()
        }
        short_q = (query_text[:60] + '...') if len(query_text) > 60 else query_text
        try:
            supabase.table("shared_llm_cache").insert(rec).execute()
            count += 1
            print(f"[SEEDED \u2705] ({count}/{SEED_COUNT}) {short_q}")
        except Exception as row_e:
            err_str = str(row_e)
            if "duplicate key" in err_str or "23505" in err_str:
                print(f"[SKIPPED \u26a0\ufe0f] Duplicate: {short_q}")
            else:
                print(f"[SKIPPED \u26a0\ufe0f] Error ({err_str[:40]}...): {short_q}")

    return count

# ---------------------------------------------------------------------------
# Main seeding loop — Dolly 15K JSONL: instruction / context / response
# ---------------------------------------------------------------------------
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)

        # ── Skip rows already processed (checkpoint mode) ──────────────────
        if i < row_offset:
            if i % 1000 == 0:
                print(f"[SKIP] Fast-forwarding... ({i}/{row_offset})")
            continue

        total_inspected += 1

        # ── ShareGPT fields ────────────────────────────────────────────────
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue

        human_turns = [c for c in convs if c.get("from", "") in ("human", "user")]
        gpt_turns   = [c for c in convs if c.get("from", "") in ("gpt", "assistant")]

        if not human_turns or not gpt_turns:
            continue

        instruction = human_turns[0].get("value", "").strip()
        response    = gpt_turns[0].get("value", "").strip()

        # Skip if essential fields are empty
        if not instruction or not response:
            continue

        # ── Length filter ───────────────────────────────────────────────────
        if len(instruction) >= MAX_QUERY_CHARS:
            continue

        # ── Dedup (first-run only, no checkpoint) ───────────────────────────
        if existing_queries and instruction in existing_queries:
            continue

        batch_queries.append(instruction)
        batch_responses.append(response)

        # ── Flush every BATCH_SIZE valid records ────────────────────────────
        if len(batch_queries) >= BATCH_SIZE:
            count = flush_batch(batch_queries, batch_responses, count)
            batch_queries.clear()
            batch_responses.clear()

            mid_checkpoint = {
                "last_inspected": row_offset + total_inspected,
                "last_inserted":  cumulative_total + count,
                "dataset":        DATASET_NAME
            }
            with open(CHECKPOINT_PATH, "w") as cp:
                json.dump(mid_checkpoint, cp, indent=2)
            print(f"[CHECKPOINT] Mid-run checkpoint saved (inspected: {mid_checkpoint['last_inspected']})")

        if count >= SEED_COUNT:
            print(f"\n[DEBUG] Reached target of {SEED_COUNT} inserts. Stopping.")
            break

# ── Flush any remaining records in the last partial batch ──────────────────
if batch_queries and count < SEED_COUNT:
    count = flush_batch(batch_queries, batch_responses, count)

# ---------------------------------------------------------------------------
# Save Final Checkpoint
# ---------------------------------------------------------------------------
new_checkpoint = {
    "last_inspected": row_offset + total_inspected,
    "last_inserted":  cumulative_total + count,
    "dataset":        DATASET_NAME
}

with open(CHECKPOINT_PATH, "w") as f:
    json.dump(new_checkpoint, f, indent=2)

print(f"\n[CHECKPOINT] Final checkpoint saved → {CHECKPOINT_PATH}")
print(f"             last_inspected : {new_checkpoint['last_inspected']}")
print(f"             last_inserted  : {new_checkpoint['last_inserted']}")
print(f"\n[DONE] Run complete! New rows inserted: {count} | Total in cache: ~{new_checkpoint['last_inserted']}")