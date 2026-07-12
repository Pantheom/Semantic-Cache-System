"""
AXIOM Semantic Cache — Accuracy Test Runner
Sends all prompts from a CSV to the local API and reports metrics.

Usage:
    python test_accuracy.py                                        # default: test.csv
    python test_accuracy.py --csv semantic_cache_test_prompts.csv  # new 600-prompt CSV
"""

import csv
import json
import time
import sys
import argparse
import requests
from collections import defaultdict

API_URL  = "http://127.0.0.1:8000/query"
TIMEOUT  = 30   # seconds per request

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"
GR = "\033[92m"   # green
BL = "\033[94m"   # blue
YL = "\033[93m"   # yellow
RD = "\033[91m"   # red
CY = "\033[96m"   # cyan
DM = "\033[90m"   # dim
MG = "\033[95m"   # magenta

# ── Helpers ───────────────────────────────────────────────────────────────────
def bar(val, total, width=36, fill="█", empty="░"):
    filled = int(round(val / total * width)) if total else 0
    return fill * filled + empty * (width - filled)

def pct(val, total):
    return val / total * 100 if total else 0

def check_server():
    try:
        r = requests.get("http://127.0.0.1:8000/docs", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def load_prompts(path):
    """
    Load prompts from a CSV. Handles both column schemas:
      - test.csv                      : prompt_id, prompt
      - semantic_cache_test_prompts.csv: id, pair_id, category, prompt
    Returns list of dicts with keys: id, text, category (may be None).
    """
    prompts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Auto-detect ID column
        id_col  = "prompt_id" if "prompt_id" in headers else "id"
        has_cat = "category" in headers

        for row in reader:
            prompts.append({
                "id":       row[id_col],
                "text":     row["prompt"],
                "category": row.get("category") if has_cat else None,
            })
    return prompts, has_cat

def send_query(prompt_text):
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            API_URL,
            json={"prompt": prompt_text},
            timeout=TIMEOUT
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        data = resp.json()
        return data, elapsed, None
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000)
        return None, elapsed, str(e)

def classify(source):
    if not source:
        return "ERROR"
    s = source.upper()
    if s.startswith("RAM"):  return "RAM"
    if s.startswith("DB"):   return "DB"
    if s.startswith("LLM"):  return "LLM"
    return "ERROR"

# ── Print helpers ─────────────────────────────────────────────────────────────
def print_header(csv_path, total):
    print()
    print(f"{B}{CY}{'═' * 68}{R}")
    print(f"{B}{CY}  AXIOM Semantic Cache — Accuracy Test Runner{R}")
    print(f"{B}{CY}{'═' * 68}{R}")
    print(f"  {DM}CSV  : {csv_path}{R}")
    print(f"  {DM}Total: {total} prompts{R}")
    print()

def print_row(idx, total, pid, prompt, source_key, latency_ms, category=None, error=None):
    badge_map = {
        "RAM":   f"{GR}[RAM HIT ]{R}",
        "DB":    f"{BL}[DB HIT  ]{R}",
        "LLM":   f"{YL}[LLM MISS]{R}",
        "ERROR": f"{RD}[ERROR   ]{R}",
    }
    badge = badge_map.get(source_key, f"{RD}[UNKNOWN ]{R}")
    short = (prompt[:48] + "…") if len(prompt) > 49 else prompt.ljust(49)
    cat_tag = f"{DM}[{category:<14}]{R} " if category else ""
    pct_str = f"{idx/total*100:5.1f}%"
    lat_str = f"{latency_ms:>5}ms"
    err_note = f"  {RD}{error[:40]}{R}" if error else ""
    print(f"  {DM}#{pid:<4}{R} {badge} {DM}{pct_str}{R}  {cat_tag}{short}  {DM}{lat_str}{R}{err_note}")

def print_summary(stats, cat_stats, total, total_ms, has_cat):
    ram   = stats["RAM"]
    db    = stats["DB"]
    llm   = stats["LLM"]
    err   = stats["ERROR"]
    hits  = ram + db
    avg_ms = total_ms / total if total else 0
    accuracy = pct(hits, total)

    print()
    print(f"{B}{CY}{'═' * 68}{R}")
    print(f"{B}{CY}  RESULTS SUMMARY{R}")
    print(f"{B}{CY}{'═' * 68}{R}")
    print()

    # ── Core metrics ──────────────────────────────────────────────────────────
    print(f"  {B}Total Prompts Tested :{R}  {total}")
    print()
    print(f"  {GR}{B}RAM Hits             :{R}  {ram:<5}  {DM}{bar(ram, total, 30)}{R}  {GR}{pct(ram, total):5.1f}%{R}")
    print(f"  {BL}{B}DB (Cache) Hits      :{R}  {db:<5}  {DM}{bar(db,  total, 30)}{R}  {BL}{pct(db,  total):5.1f}%{R}")
    print(f"  {YL}{B}Misses               :{R}  {llm:<5}  {DM}{bar(llm, total, 30)}{R}  {YL}{pct(llm, total):5.1f}%{R}")
    if err:
        print(f"  {RD}{B}Errors               :{R}  {err}")
    print()
    print(f"  {'─' * 50}")
    print(f"  {B}Accuracy  (Hits / Total Prompts) :{R}  {B}{CY}{accuracy:.1f}%{R}  {DM}({hits}/{total}){R}")
    print(f"  {B}Avg Latency per Query            :{R}  {avg_ms:.0f} ms")
    print(f"  {B}Total Test Duration              :{R}  {total_ms/1000:.1f}s")
    print()

    # ── Per-category breakdown (only when category column is present) ──────────
    if has_cat and cat_stats:
        print(f"  {'─' * 50}")
        print(f"  {B}{MG}Per-Category Breakdown{R}")
        print(f"  {'─' * 50}")
        # Sort by hit rate descending
        sorted_cats = sorted(
            cat_stats.items(),
            key=lambda x: pct(x[1]["RAM"] + x[1]["DB"], x[1]["total"]),
            reverse=True
        )
        for cat, s in sorted_cats:
            c_total = s["total"]
            c_hits  = s["RAM"] + s["DB"]
            c_ram   = s["RAM"]
            c_db    = s["DB"]
            c_miss  = s["LLM"]
            c_acc   = pct(c_hits, c_total)
            c_bar   = bar(c_hits, c_total, 20)
            print(
                f"  {MG}{cat:<18}{R}  "
                f"{DM}{c_bar}{R}  "
                f"{B}{CY}{c_acc:5.1f}%{R}  "
                f"{DM}({c_total} prompts | "
                f"{GR}RAM:{c_ram}{DM} "
                f"{BL}DB:{c_db}{DM} "
                f"{YL}MISS:{c_miss}{DM}){R}"
            )
        print()

    print(f"{B}{CY}{'═' * 68}{R}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXIOM Semantic Cache — Accuracy Test Runner")
    parser.add_argument(
        "--csv",
        default="test.csv",
        help="Path to the CSV file to test (default: test.csv)"
    )
    args = parser.parse_args()
    csv_path = args.csv

    # Server check
    print(f"\n  Checking API at {API_URL} …", end=" ", flush=True)
    if not check_server():
        print(f"{RD}OFFLINE{R}")
        print(f"\n  {RD}ERROR:{R} uvicorn is not running. Start it with:")
        print(f"          uvicorn main:app --reload\n")
        sys.exit(1)
    print(f"{GR}ONLINE{R}")

    # Load CSV
    try:
        prompts, has_cat = load_prompts(csv_path)
    except FileNotFoundError:
        print(f"\n  {RD}ERROR:{R} '{csv_path}' not found. Run from the semantic_cache directory.\n")
        sys.exit(1)

    total    = len(prompts)
    stats    = defaultdict(int)
    cat_stats = defaultdict(lambda: defaultdict(int))  # category → source → count
    total_ms = 0

    print_header(csv_path, total)
    print(f"  {B}Running {total} prompts…{R}\n")
    print(f"  {'─' * 66}")

    for idx, p in enumerate(prompts, 1):
        data, elapsed, error = send_query(p["text"])
        total_ms += elapsed

        if error or data is None:
            src_key = "ERROR"
        else:
            src_key = classify(data.get("source"))

        stats[src_key] += 1

        # Track per-category stats
        cat = p["category"]
        if cat:
            cat_stats[cat]["total"] += 1
            cat_stats[cat][src_key] += 1

        print_row(idx, total, p["id"], p["text"], src_key, elapsed,
                  category=cat, error=error if src_key == "ERROR" else None)

        # Live progress every 50 rows
        if idx % 50 == 0:
            hits_so_far = stats["RAM"] + stats["DB"]
            running_pct = pct(hits_so_far, idx)
            print(f"\n  {DM}── Progress {idx}/{total} │ Running hit rate: {running_pct:.1f}% ──{R}\n")

    print_summary(stats, cat_stats, total, total_ms, has_cat)
