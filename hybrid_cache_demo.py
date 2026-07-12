from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import csv
import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import spacy
from spacy.cli import download as spacy_download
from sentence_transformers import SentenceTransformer

try:
    import ollama
except ImportError:
    print("OLLAMA NOT FOUND")
    ollama = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("RANK_BM25 NOT found")
    BM25Okapi = None

MODEL_NAME = "all-mpnet-base-v2"
DIM = 768
TOP_K_DENSE = 5
TOP_K_BM25 = 5

DEFAULT_THRESHOLD = 0.68

THRESHOLD_BY_TYPE = {
    "general": 0.62,
    "coding": 0.72,
    "factual": 0.68,
    "chat": 0.64,
}

DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35
FINAL_WEIGHT_HYBRID = 0.68
FINAL_WEIGHT_METADATA = 0.12
FINAL_WEIGHT_VALIDATOR = 0.30

THRESHOLD_RELAX_STRONG_METADATA = 0.02
THRESHOLD_RELAX_MODERATE_METADATA = 0.01
THRESHOLD_RELAX_VALIDATOR_OK = 0.04
THRESHOLD_RELAX_HIGH_HYBRID = 0.01
THRESHOLD_MAX_RELAX = 0.08
THRESHOLD_FLOOR = 0.58
ENTITY_CONFLICT_HARD_REJECT_HYBRID = 0.45
ENTITY_CONFLICT_PENALTY_FACTOR = 0.55
METADATA_OVERLAP_STRONG = 0.25
METADATA_OVERLAP_MODERATE = 0.10
METADATA_OVERLAP_WEAK = 0.0
BM25_NORM_EPS = 1e-8
VALIDATOR_TOP_K = 3

VALIDATOR_MIN_HYBRID = 0.45
VALIDATOR_MAX_HYBRID = 0.99
VALIDATOR_MIN_CONFIDENCE = 0.50
VALIDATOR_TIMEOUT_SEC = 10.0
VALIDATOR_MAX_RETRIES = 2
VALIDATOR_RETRY_BACKOFF_SEC = 0.25
VALIDATOR_HARD_REJECT_CONF = 0.85
VALIDATOR_SOFT_PENALTY = -0.10
TECHNICAL_FAILURE_STATUSES = {
    "timeout",
    "error",
    "empty_response",
    "invalid_json",
    "exception",
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "tinyllama"

FILE_ID = "14sRntjjOJ6Ghp2_Qo_uTf6-PCZqOeRrt"
DEFAULT_CSV_PATH = Path(__file__).with_name("test.csv")

NEGATION_WORDS = {
    "no",
    "not",
    "without",
    "except",
    "exclude",
    "excluding",
}

COUNTRY_ALIAS_MAP = {
    "america": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "unitedstates": "usa",
    "unitedstatesofamerica": "usa",
    "us": "usa",
    "u.s": "usa",
    "u.s.": "usa",
    "u.s.a": "usa",
    "u.s.a.": "usa",
    "usa": "usa",
    "uk": "united_kingdom",
    "u.k": "united_kingdom",
    "u.k.": "united_kingdom",
    "united kingdom": "united_kingdom",
    "unitedkingdom": "united_kingdom",
    "britain": "united_kingdom",
    "great britain": "united_kingdom",
    "greatbritain": "united_kingdom",
    "england": "united_kingdom",
    "france": "france",
    "japan": "japan",
    "canada": "canada",
    "mexico": "mexico",
    "china": "china",
    "india": "india",
    "germany": "germany",
    "italy": "italy",
    "spain": "spain",
    "russia": "russia",
    "brazil": "brazil",
    "australia": "australia",
}

COUNTRY_CONFLICT_PENALTY = 0.25
COUNTRY_PARTIAL_PENALTY = 0.92
COUNTRY_RERANK_MATCH_BONUS = 0.10
COUNTRY_RERANK_MISMATCH_PENALTY = 0.10
COUNTRY_RERANK_MAX_DELTA = 0.12

_model = None
_nlp = None


@dataclasses.dataclass
class ExperimentMetrics:
    """Holds all metrics for a single cache experiment run."""
    label: str
    total: int
    hits_accepted: int
    hits_rejected: int
    misses: int
    hit_rate: float
    avg_score: float
    reuse_accepted: int
    reuse_rejected: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


def log_debug(message: str, debug: bool) -> None:
    if debug:
        print(f"[DEBUG] {message}")


class CsvTee:
    def __init__(self, orig, path: Path):
        self._orig = orig
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

    def write(self, text: str) -> None:
        try:
            self._orig.write(text)
        except Exception:
            pass
        if not text:
            return
        for line in text.rstrip("\n").split("\n"):
            if line.strip():
                self._writer.writerow([time.time(), line])
        try:
            self._file.flush()
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False


def enable_csv_output(path: Path) -> None:
    tee_out = CsvTee(sys.stdout, path)
    tee_err = CsvTee(sys.stderr, path)
    sys.stdout = tee_out
    sys.stderr = tee_err


def load_models(debug: bool = False) -> None:
    global _model, _nlp
    log_info("Loading embedding model...")
    _model = SentenceTransformer(MODEL_NAME)
    log_debug(f"Embedding model ready: {MODEL_NAME}", debug)
    log_info("Loading spaCy model...")
    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        log_warn("spaCy model en_core_web_sm not found, downloading...")
        spacy_download("en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
    log_debug("spaCy model ready: en_core_web_sm", debug)
    log_info("Models loaded")


def embed(text: str) -> np.ndarray:
    vec = _model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def classify_intent(text: str, doc=None) -> str:
    """Lightweight rule-based intent classification.

    Returns one of: coding, factual, explanation, comparison, procedural, general
    """
    if doc is None:
        try:
            doc = _nlp(text.lower())
        except Exception:
            doc = None

    lower = text.lower()
    # coding / implementation intent
    if re.search(r"\b(how to|how do i|implement|write code|example|function\b|class\b|def\b)", lower):
        return "coding"
    # explicit procedural how-to
    if re.search(r"\b(how to|step by step|steps to|procedure|install|setup)\b", lower):
        return "procedural"
    # comparison / difference
    if re.search(r"\b(compare|difference|vs\b|versus|contrast)\b", lower):
        return "comparison"
    # ask-for-explanation
    if re.search(r"\b(explain|why|how come)\b", lower):
        return "explanation"
    # factual short answer questions
    if lower.strip().startswith(("what", "who", "when", "where", "which")) or "?" in lower:
        return "factual"

    return "general"


def extract_metadata(text: str) -> dict:
    doc = _nlp(text.lower())

    entities = []
    nouns = []
    negations = []
    numbers = []

    tokens = [t.text for t in doc]

    for ent in doc.ents:
        entities.append(ent.text)

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            nouns.append(token.lemma_)

    for token in doc:
        if token.like_num:
            numbers.append(token.text)

    for i, token in enumerate(tokens):
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            negations.append(tokens[i + 1])

    root_verb = ""
    for token in doc:
        if token.dep_ == "ROOT":
            root_verb = token.lemma_
            break

    noun_sig = "-".join(sorted(set(nouns[:3])))
    intent_class = classify_intent(text, doc)
    intent = f"{intent_class}:{root_verb}:{noun_sig}"

    return {
        "intent": intent,
        "intent_class": intent_class,
        "entities": sorted(set(entities)),
        "nouns": sorted(set(nouns)),
        "negations": sorted(set(negations)),
        "numbers": sorted(set(numbers)),
        "source_text": text,
    }


def _overlap_ratio(left: set, right: set) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _strong_entity_terms(entities: list[str]) -> set[str]:
    terms: set[str] = set()
    for ent in entities or []:
        for token in re.findall(r"[a-z0-9_]+", ent.lower()):
            if len(token) >= 3 and token not in NEGATION_WORDS:
                terms.add(token)
    return terms


def _negation_targets(meta: dict) -> set[str]:
    negations = set(meta.get("negations", []))
    if not negations:
        return set()
    nouns = set(meta.get("nouns", []))
    entities = _strong_entity_terms(meta.get("entities", []))
    return {term for term in negations if term in nouns or term in entities}


def normalize_country_text(text: str) -> list[str]:
    if not text:
        return []
    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    return cleaned.split()


def _build_country_alias_tokens() -> dict[tuple[str, ...], str]:
    tokens_map: dict[tuple[str, ...], str] = {}
    for alias, canonical in COUNTRY_ALIAS_MAP.items():
        tokens = tuple(normalize_country_text(alias))
        if not tokens:
            continue
        tokens_map[tokens] = canonical
    return tokens_map


COUNTRY_ALIAS_TOKENS = _build_country_alias_tokens()
COUNTRY_ALIAS_MAX_TOKENS = max((len(key) for key in COUNTRY_ALIAS_TOKENS), default=1)


def extract_countries(text: str) -> set[str]:
    tokens = normalize_country_text(text)
    if not tokens:
        return set()
    found = set()
    max_len = COUNTRY_ALIAS_MAX_TOKENS
    for i in range(len(tokens)):
        for size in range(1, max_len + 1):
            if i + size > len(tokens):
                break
            window = tuple(tokens[i : i + size])
            canonical = COUNTRY_ALIAS_TOKENS.get(window)
            if canonical:
                found.add(canonical)
    return found


def _extract_country_entities(meta: dict) -> set[str]:
    source_text = meta.get("source_text", "") or ""
    countries = extract_countries(source_text)
    if countries:
        return countries
    fallback = " ".join(list(meta.get("entities", [])) + list(meta.get("nouns", [])))
    return extract_countries(fallback)


def metadata_decision(query_meta: dict, cached_meta: dict) -> str:
    q_neg = _negation_targets(query_meta)
    c_neg = _negation_targets(cached_meta)
    if q_neg or c_neg:
        if not q_neg or not c_neg:
            return "mismatch"
        if q_neg.isdisjoint(c_neg):
            return "mismatch"

    q_nums = set(query_meta.get("numbers", []))
    c_nums = set(cached_meta.get("numbers", []))
    if q_nums and c_nums and q_nums.isdisjoint(c_nums):
        return "mismatch"

    q_nouns = set(query_meta.get("nouns", []))
    c_nouns = set(cached_meta.get("nouns", []))
    overlap = _overlap_ratio(q_nouns, c_nouns)

    if overlap is None:
        return "uncertain"

    if overlap >= 0.35:
        return "match"
    if overlap >= 0.10:
        return "uncertain"

    return "uncertain"


def metadata_match(query_meta: dict, cached_meta: dict) -> bool:
    return metadata_decision(query_meta, cached_meta) == "match"


def metadata_label(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.60:
        return "moderate"
    if score >= 0.30:
        return "weak"
    return "low"


class TinyLlamaValidator:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        timeout_sec: float = VALIDATOR_TIMEOUT_SEC,
        min_confidence: float = VALIDATOR_MIN_CONFIDENCE,
    ):
        self.model = model
        self.timeout_sec = timeout_sec
        self.min_confidence = min_confidence
        self.max_retries = VALIDATOR_MAX_RETRIES
        self.retry_backoff_sec = VALIDATOR_RETRY_BACKOFF_SEC
        self._memo = {}
        if ollama is None:
            raise RuntimeError("ollama library is not installed. Run: pip install ollama")
        self._ollama_client = ollama.Client(host=OLLAMA_HOST)

    def _build_prompt(self, cached_query: str, incoming_query: str) -> str:
        return (
            "You are an EXTREMELY STRICT semantic cache validator.\n"
            "\n"
            "Your ONLY task:\n"
            "Decide whether the EXACT SAME answer can fully answer BOTH queries.\n"
            "\n"
            "IMPORTANT:\n"
            "- Same topic is NOT enough\n"
            "- Same field/domain is NOT enough\n"
            "- Related concepts are NOT enough\n"
            "- Similar wording is NOT enough\n"
            "- The exact same response must satisfy both users equally well\n"
            "- If one query needs additional explanation, return false\n"
            "- If unsure, ALWAYS return false\n"
            "\n"
            "GOOD REUSE:\n"
            "Q1: What is overfitting?\n"
            "Q2: Define overfitting in machine learning.\n"
            "Answer: true\n"
            "\n"
            "Q1: What is the capital of France?\n"
            "Q2: What's France's capital city?\n"
            "Answer: true\n"
            "\n"
            "BAD REUSE:\n"
            "Q1: Explain k-means clustering.\n"
            "Q2: Explain EM algorithm.\n"
            "Answer: false\n"
            "\n"
            "Q1: What is classification?\n"
            "Q2: What is regression?\n"
            "Answer: false\n"
            "\n"
            "Q1: What is hallucination detection?\n"
            "Q2: What is a context window?\n"
            "Answer: false\n"
            "\n"
            "Q1: What is a feature store?\n"
            "Q2: What is a virtual environment?\n"
            "Answer: false\n"
            "\n"
            "Return ONLY valid JSON.\n"
            "Format:\n"
            "{\"safe_reuse\": true, \"confidence\": 0.95}\n"
            "\n"
            f"Cached: {cached_query}\n"
            f"Incoming: {incoming_query}\n"
        )

    def _sanitize_response(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\x00", "").strip()
        cleaned = re.sub(r"^```[a-z0-9]*\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _display_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _find_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _extract_json(self, text: str):
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            candidate = self._find_json_object(text)
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
        return None

    def _generate_request(self, prompt: str, options: dict, format_json: bool) -> dict:
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if format_json:
            kwargs["format"] = "json"
        return self._ollama_client.generate(**kwargs)

    def _generate_with_timeout(self, prompt: str, options: dict, format_json: bool) -> dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._generate_request, prompt, options, format_json)
            try:
                return future.result(timeout=self.timeout_sec)
            except concurrent.futures.TimeoutError:
                return {"error": "timeout"}
            except Exception as exc:
                return {"error": str(exc)}

    def _normalize_result(self, data):
        if not isinstance(data, dict):
            return None
        safe = data.get("safe_reuse")
        conf = data.get("confidence")
        if not isinstance(safe, bool):
            return None
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            return None
        conf = max(0.0, min(1.0, conf))
        return {"safe_reuse": safe, "confidence": conf}

    def validate(self, cached_query: str, incoming_query: str, debug: bool = False) -> dict:
        key = (cached_query, incoming_query)
        if key in self._memo:
            log_debug("Validator memo hit", debug)
            return self._memo[key]

        prompt = self._build_prompt(cached_query, incoming_query)
        options = {
            "temperature": 0,
            "top_k": 1,
            "top_p": 0.1,
            "num_predict": 60,
        }

        result = {"safe_reuse": False, "confidence": 0.0, "status": "error"}
        last_status = "error"
        try:
            log_info("TINYLLAMA VALIDATION")
            print("-" * 60)
            print(f"  Cached   : {cached_query}")
            print(f"  Incoming : {incoming_query}")
            print("-" * 60)
            log_debug(f"Using ollama client host={OLLAMA_HOST}", debug)

            for attempt in range(self.max_retries + 1):
                use_json_format = attempt == 0
                log_debug(
                    f"Validator attempt {attempt + 1}/{self.max_retries + 1} format_json={use_json_format}",
                    debug,
                )
                outer = self._generate_with_timeout(prompt, options, use_json_format)
                response_text = self._sanitize_response(outer.get("response", ""))
                display_text = self._display_text(response_text)

                if response_text:
                    print(f"TinyLlama response - {display_text}")
                else:
                    print("TinyLlama response - <EMPTY RESPONSE>")

                if "error" in outer:
                    last_status = "timeout" if outer["error"] == "timeout" else "error"
                    log_warn(f"TinyLlama error: {outer['error']}")
                    if debug:
                        print(json.dumps(outer, indent=2, ensure_ascii=True))
                elif not response_text:
                    last_status = "empty_response"
                    log_warn("TinyLlama returned empty response")
                else:
                    parsed = self._extract_json(response_text)
                    normalized = self._normalize_result(parsed)
                    if normalized is not None:
                        result = {"status": "ok", **normalized}
                        last_status = "ok"
                        sys.stdout.flush()
                        break
                    last_status = "invalid_json"
                    log_warn("TinyLlama response malformed or missing fields")

                sys.stdout.flush()
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec)

            if last_status != "ok":
                result = {
                    "safe_reuse": False,
                    "confidence": 0.0,
                    "status": last_status,
                }
        except Exception as exc:
            result = {"safe_reuse": False, "confidence": 0.0, "status": "exception", "error": str(exc)}

        log_debug(f"Validator result: {result}", debug)

        self._memo[key] = result
        if len(self._memo) > 512:
            self._memo.clear()
        return result


class HybridSemanticCache:
    def __init__(
        self,
        dim: int = DIM,
        top_k_dense: int = TOP_K_DENSE,
        top_k_bm25: int = TOP_K_BM25,
        default_threshold: float = DEFAULT_THRESHOLD,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        validator: TinyLlamaValidator | None = None,
        validator_min_hybrid: float = VALIDATOR_MIN_HYBRID,
        validator_max_hybrid: float = VALIDATOR_MAX_HYBRID,
    ):
        self.dim = dim
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.default_threshold = default_threshold
        self.dense_weight, self.bm25_weight = self._normalize_weights(dense_weight, bm25_weight)
        self.final_weight_hybrid, self.final_weight_metadata, self.final_weight_validator = (
            self._normalize_final_weights(
                FINAL_WEIGHT_HYBRID,
                FINAL_WEIGHT_METADATA,
                FINAL_WEIGHT_VALIDATOR,
            )
        )
        self.validator = validator
        self.validator_min_hybrid = validator_min_hybrid
        self.validator_max_hybrid = validator_max_hybrid
        self.index = faiss.IndexFlatIP(dim)
        self.entries = []
        self._bm25 = None
        self._bm25_corpus = []
        self._exact = {}
        print(f"[CACHE] Hybrid cache initialized dim={dim}")

    def _normalize_weights(self, dense_weight: float, bm25_weight: float) -> tuple[float, float]:
        total = dense_weight + bm25_weight
        if total <= 0:
            return 0.65, 0.35
        return dense_weight / total, bm25_weight / total

    def _normalize_final_weights(
        self,
        hybrid_weight: float,
        metadata_weight: float,
        validator_weight: float,
    ) -> tuple[float, float, float]:
        total = hybrid_weight + metadata_weight + validator_weight
        if total <= 0:
            return 0.85, 0.10, 0.05
        return hybrid_weight / total, metadata_weight / total, validator_weight / total

    def _final_threshold(
        self,
        response_type: str,
        metadata_score: float,
        validator_status: str,
        hybrid_score: float,
        validator_safe_reuse: bool | None = None,
        validator_confidence: float | None = None,
    ) -> tuple[float, dict]:
        base = THRESHOLD_BY_TYPE.get(response_type, self.default_threshold)
        relax = 0.0
        if metadata_score >= 0.85:
            relax += THRESHOLD_RELAX_STRONG_METADATA
        elif metadata_score >= 0.60:
            relax += THRESHOLD_RELAX_MODERATE_METADATA
        if validator_safe_reuse:
            conf = validator_confidence or 0.0
            if conf >= 0.90:
                relax += 0.08
            elif conf >= 0.80:
                relax += 0.05
            else:
                relax += 0.03
        if hybrid_score >= 0.90:
            relax += THRESHOLD_RELAX_HIGH_HYBRID
        relax = min(relax, THRESHOLD_MAX_RELAX)
        final = max(THRESHOLD_FLOOR, base - relax)
        return final, {"base": base, "relax": relax, "final": final}

    def _country_rerank_adjustment(self, query_meta: dict, cached_meta: dict) -> tuple[float, dict]:
        q_countries = _extract_country_entities(query_meta)
        c_countries = _extract_country_entities(cached_meta)

        if not q_countries or not c_countries:
            return 0.0, {
                "q_countries": sorted(q_countries),
                "c_countries": sorted(c_countries),
                "country_adjustment": 0.0,
                "country_reason": "insufficient_country_signals",
            }

        if q_countries & c_countries:
            delta = COUNTRY_RERANK_MATCH_BONUS
            reason = "country_match"
        else:
            delta = -COUNTRY_RERANK_MISMATCH_PENALTY
            reason = "country_mismatch"

        delta = max(-COUNTRY_RERANK_MAX_DELTA, min(COUNTRY_RERANK_MAX_DELTA, delta))
        return delta, {
            "q_countries": sorted(q_countries),
            "c_countries": sorted(c_countries),
            "country_adjustment": delta,
            "country_reason": reason,
        }

    def _has_hard_contradiction(self, query_meta: dict, cached_meta: dict) -> tuple[bool, str | None, dict]:
        q_neg = _negation_targets(query_meta)
        c_neg = _negation_targets(cached_meta)
        if q_neg or c_neg:
            if not q_neg or not c_neg:
                return True, "negation_mismatch", {"q_neg": sorted(q_neg), "c_neg": sorted(c_neg)}
            if q_neg.isdisjoint(c_neg):
                return True, "negation_mismatch", {"q_neg": sorted(q_neg), "c_neg": sorted(c_neg)}

        q_nums = set(query_meta.get("numbers", []))
        c_nums = set(cached_meta.get("numbers", []))
        if q_nums and c_nums and q_nums.isdisjoint(c_nums):
            return True, "number_mismatch", {"q_nums": sorted(q_nums), "c_nums": sorted(c_nums)}

        return False, None, {}

    def _metadata_score(
        self,
        query_meta: dict,
        cached_meta: dict,
        hybrid_score: float,
    ) -> tuple[float, bool, dict]:
        hard_contradiction, reason, contradiction_details = self._has_hard_contradiction(
            query_meta, cached_meta
        )
        if hard_contradiction:
            details = {"reason": reason}
            details.update(contradiction_details)
            return 0.0, True, details

        q_nouns = set(query_meta.get("nouns", []))
        c_nouns = set(cached_meta.get("nouns", []))
        overlap = _overlap_ratio(q_nouns, c_nouns)

        if overlap is None:
            base = 0.65
            overlap_bucket = "none"
        elif overlap >= METADATA_OVERLAP_STRONG:
            base = 1.00
            overlap_bucket = "strong"
        elif overlap >= METADATA_OVERLAP_MODERATE:
            base = 0.75
            overlap_bucket = "moderate"
        elif overlap > METADATA_OVERLAP_WEAK:
            base = 0.55
            overlap_bucket = "weak"
        else:
            base = 0.45
            overlap_bucket = "none"

        q_ents = _strong_entity_terms(query_meta.get("entities", []))
        c_ents = _strong_entity_terms(cached_meta.get("entities", []))
        entity_conflict = bool(q_ents and c_ents and q_ents.isdisjoint(c_ents))
        if entity_conflict and hybrid_score < ENTITY_CONFLICT_HARD_REJECT_HYBRID:
            return 0.0, True, {
                "reason": "entity_conflict_low_hybrid",
                "q_entities": sorted(q_ents),
                "c_entities": sorted(c_ents),
            }

        if not q_ents and not c_ents:
            entity_factor = 1.0
        elif entity_conflict:
            entity_factor = ENTITY_CONFLICT_PENALTY_FACTOR
        elif q_ents & c_ents:
            entity_factor = 0.92
        else:
            entity_factor = 0.96

        q_countries = _extract_country_entities(query_meta)
        c_countries = _extract_country_entities(cached_meta)
        country_conflict = False
        if q_countries and c_countries:
            if q_countries.isdisjoint(c_countries):
                country_conflict = True
                country_factor = COUNTRY_CONFLICT_PENALTY
            else:
                country_factor = 1.0
        elif q_countries or c_countries:
            country_factor = COUNTRY_PARTIAL_PENALTY
        else:
            country_factor = 1.0

        q_nums = set(query_meta.get("numbers", []))
        c_nums = set(cached_meta.get("numbers", []))
        number_factor = 1.0
        if bool(q_nums) ^ bool(c_nums):
            number_factor = 0.92

        q_neg_raw = set(query_meta.get("negations", []))
        c_neg_raw = set(cached_meta.get("negations", []))
        q_neg = _negation_targets(query_meta)
        c_neg = _negation_targets(cached_meta)
        negation_factor = 1.0
        if q_neg_raw or c_neg_raw:
            if not q_neg and not c_neg:
                negation_factor = 0.97
            else:
                negation_factor = 0.95

        score = min(
            1.0,
            base * entity_factor * number_factor * negation_factor * country_factor,
        )
        details = {
            "overlap": overlap,
            "base": base,
            "overlap_bucket": overlap_bucket,
            "entity_factor": entity_factor,
            "entity_conflict": entity_conflict,
            "entity_penalty_applied": entity_conflict and hybrid_score >= ENTITY_CONFLICT_HARD_REJECT_HYBRID,
            "number_factor": number_factor,
            "negation_factor": negation_factor,
            "country_factor": country_factor,
            "country_conflict": country_conflict,
            "q_countries": sorted(q_countries),
            "c_countries": sorted(c_countries),
            "q_entities": sorted(q_ents),
            "c_entities": sorted(c_ents),
            "q_nums": sorted(q_nums),
            "c_nums": sorted(c_nums),
            "q_neg": sorted(q_neg),
            "c_neg": sorted(c_neg),
        }
        return score, False, details

    def _norm(self, vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return (vec / n).astype(np.float32) if n > 0 else vec.astype(np.float32)

    def _rebuild_bm25(self) -> None:
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 is not installed. Run: pip install rank_bm25")
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def _record_retrieval(self, entry: dict, accepted: bool, result: dict) -> None:
        now = time.time()
        entry["last_accessed"] = now
        entry["access_count"] += 1
        if accepted:
            entry["hit_count"] += 1
        else:
            entry["reject_count"] += 1
        entry["last_score"] = result.get("score")
        entry["last_final_score"] = result.get("final_score")
        entry["last_hybrid_score"] = result.get("hybrid_score")
        entry["last_metadata_score"] = result.get("metadata_score")
        entry["last_validator_status"] = result.get("validator_status")
        entry["last_validator_confidence"] = result.get("validator_confidence")

    def store(
        self,
        query: str,
        response: str,
        embedding: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> None:
        if embedding is None:
            embedding = embed(query)
        if metadata is None:
            metadata = extract_metadata(query)

        normed = self._norm(embedding).reshape(1, -1)
        self.index.add(normed)

        self.entries.append(
            {
                "query": query,
                "response": response,
                "embedding": embedding,
                "metadata": metadata,
                "created_at": time.time(),
                "last_accessed": None,
                "access_count": 0,
                "hit_count": 0,
                "reject_count": 0,
                "last_score": None,
                "last_final_score": None,
                "last_hybrid_score": None,
                "last_metadata_score": None,
                "last_validator_status": None,
                "last_validator_confidence": None,
            }
        )

        self._exact[query] = len(self.entries) - 1
        self._bm25_corpus.append(tokenize(query))
        self._rebuild_bm25()

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self.entries), dtype=np.float32)
        scores = self._bm25.get_scores(tokenize(query))
        return np.asarray(scores, dtype=np.float32)

    def _normalize_bm25(self, scores: np.ndarray, candidate_indices: list[int]) -> dict[int, float]:
        if not candidate_indices:
            return {}
        values = [float(scores[i]) for i in candidate_indices]
        if not values:
            return {i: 0.0 for i in candidate_indices}
        min_score = min(values)
        max_score = max(values)
        denom = (max_score - min_score) + BM25_NORM_EPS
        return {i: (float(scores[i]) - min_score) / denom for i in candidate_indices}

    def _dense_candidates(self, embedding: np.ndarray) -> dict[int, float]:
        if self.index.ntotal == 0:
            return {}
        normed = self._norm(embedding).reshape(1, -1)
        scores, indices = self.index.search(normed, self.top_k_dense)
        scores = scores[0]
        indices = indices[0]
        return {int(i): float(s) for s, i in zip(scores, indices) if i != -1}

    def _bm25_candidates(self, query: str) -> tuple[dict[int, float], np.ndarray]:
        if not self.entries:
            return {}, np.zeros(0, dtype=np.float32)
        scores = self._bm25_scores(query)
        if scores.size == 0:
            return {}, scores
        top_idx = np.argsort(scores)[::-1][: self.top_k_bm25]
        return {int(i): float(scores[i]) for i in top_idx}, scores

    def search(
        self,
        query: str,
        embedding: np.ndarray | None = None,
        metadata: dict | None = None,
        response_type: str = "general",
        verbose: bool = False,
        debug: bool = False,
    ) -> dict:
        if query in self._exact:
            entry = self.entries[self._exact[query]]
            result = {
                "hit": True,
                "score": 1.0,
                "final_score": 1.0,
                "hybrid_score": 1.0,
                "metadata_score": 1.0,
                "metadata_decision": "strong",
                "response": entry["response"],
                "matched_query": entry["query"],
                "validator_used": False,
                "validator_confidence": None,
                "validator_safe_reuse": None,
                "validator_status": "skipped",
                "dense_score": 1.0,
                "bm25_score": 1.0,
            }
            self._record_retrieval(entry, True, result)
            return result

        if self.index.ntotal == 0:
            return {
                "hit": False,
                "score": 0.0,
                "final_score": 0.0,
                "hybrid_score": 0.0,
                "metadata_score": 0.0,
                "metadata_decision": None,
                "response": None,
                "matched_query": None,
                "validator_used": False,
                "validator_confidence": None,
                "validator_safe_reuse": None,
                "validator_status": None,
                "dense_score": 0.0,
                "bm25_score": 0.0,
            }

        if embedding is None:
            embedding = embed(query)
        if metadata is None:
            metadata = extract_metadata(query)

        dense = self._dense_candidates(embedding)
        bm25_top, bm25_scores = self._bm25_candidates(query)
        candidate_indices = list(set(dense.keys()) | set(bm25_top.keys()))

        if debug:
            dense_preview = sorted(dense.items(), key=lambda item: item[1], reverse=True)[:5]
            bm25_preview = sorted(bm25_top.items(), key=lambda item: item[1], reverse=True)[:5]
            log_debug(f"Dense candidates: {dense_preview}", debug)
            log_debug(f"BM25 candidates: {bm25_preview}", debug)

        bm25_norm = self._normalize_bm25(bm25_scores, candidate_indices)
        if debug:
            raw_pairs = sorted(
                [(int(i), float(bm25_scores[i])) for i in candidate_indices],
                key=lambda item: item[1],
                reverse=True,
            )
            norm_pairs = sorted(
                [(int(i), float(bm25_norm.get(i, 0.0))) for i in candidate_indices],
                key=lambda item: item[1],
                reverse=True,
            )
            values = [v for _, v in raw_pairs]
            if values:
                log_debug(
                    "BM25 raw range: min=%.4f max=%.4f eps=%.1e" %
                    (min(values), max(values), BM25_NORM_EPS),
                    debug,
                )
            log_debug(f"BM25 raw scores: {raw_pairs}", debug)
            log_debug(f"BM25 normalized: {norm_pairs}", debug)
        fused = {}
        for idx in candidate_indices:
            dense_score = max(0.0, min(1.0, dense.get(idx, 0.0)))
            bm25_score = bm25_norm.get(idx, 0.0)
            fused[idx] = self.dense_weight * dense_score + self.bm25_weight * bm25_score

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)

        if debug:
            fused_preview = ranked[:5]
            log_debug(f"Hybrid ranked: {fused_preview}", debug)

        best_score = 0.0
        best_entry = None
        if ranked:
            best_idx = ranked[0][0]
            best_score = float(ranked[0][1])
            best_entry = self.entries[best_idx]

        candidate_scores = []
        for idx, hybrid_score in ranked:
            entry = self.entries[idx]
            dense_score = dense.get(idx, 0.0)
            bm25_score = bm25_norm.get(idx, 0.0)

            metadata_score, hard_contradiction, meta_details = self._metadata_score(
                metadata,
                entry["metadata"],
                hybrid_score,
            )
            if hard_contradiction:
                log_warn("[METADATA] Hard rejection reason=%s" % meta_details.get("reason", "unknown"))
                log_debug(
                    "Metadata score=%.2f hard_contradiction=True details=%s" %
                    (metadata_score, meta_details),
                    debug,
                )
                continue

            if meta_details.get("entity_penalty_applied"):
                log_debug(
                    "[METADATA] Soft entity penalty factor=%.2f hybrid=%.4f" %
                    (ENTITY_CONFLICT_PENALTY_FACTOR, hybrid_score),
                    debug,
                )
            if meta_details.get("country_conflict"):
                log_debug(
                    "[METADATA] Country conflict penalty factor=%.2f q=%s c=%s" %
                    (
                        meta_details.get("country_factor", 1.0),
                        meta_details.get("q_countries", []),
                        meta_details.get("c_countries", []),
                    ),
                    debug,
                )
            elif meta_details.get("country_factor", 1.0) < 1.0:
                log_debug(
                    "[METADATA] Country mismatch soft penalty factor=%.2f q=%s c=%s" %
                    (
                        meta_details.get("country_factor", 1.0),
                        meta_details.get("q_countries", []),
                        meta_details.get("c_countries", []),
                    ),
                    debug,
                )

            log_debug(
                "Metadata score=%.2f hard_contradiction=False details=%s" %
                (metadata_score, meta_details),
                debug,
            )
            country_adjustment, country_details = self._country_rerank_adjustment(
                metadata,
                entry["metadata"],
            )
            if debug and country_adjustment != 0.0:
                log_debug(
                    "[METADATA] Country rerank adjustment=%.2f reason=%s q=%s c=%s" %
                    (
                        country_adjustment,
                        country_details.get("country_reason", "unknown"),
                        country_details.get("q_countries", []),
                        country_details.get("c_countries", []),
                    ),
                    debug,
                )

            base_score = (
                self.final_weight_hybrid * hybrid_score
                + self.final_weight_metadata * metadata_score
                + country_adjustment
            )

            if debug:
                hybrid_contrib = self.final_weight_hybrid * hybrid_score
                metadata_contrib = self.final_weight_metadata * metadata_score
                log_debug(
                    "Initial candidate idx=%s dense=%.4f bm25=%.4f hybrid=%.4f metadata=%.2f base=%.4f" %
                    (
                        idx,
                        dense_score,
                        bm25_score,
                        hybrid_score,
                        metadata_score,
                        base_score,
                    ),
                    debug,
                )
                log_debug(
                    "Initial contrib hybrid=%.4f metadata=%.4f" %
                    (hybrid_contrib, metadata_contrib),
                    debug,
                )

            candidate_scores.append(
                {
                    "idx": idx,
                    "base_score": float(base_score),
                    "final_score": float(base_score),
                    "hybrid_score": float(hybrid_score),
                    "dense_score": float(dense_score),
                    "bm25_score": float(bm25_score),
                    "metadata_score": float(metadata_score),
                    "metadata_label": metadata_label(metadata_score),
                    "country_adjustment": float(country_adjustment),
                    "country_details": country_details,
                    "validator_used": False,
                    "validator_confidence": 0.0,
                    "validator_safe_reuse": None,
                    "validator_bonus": 0.0,
                    "validator_status": "skipped",
                    "fail_open": False,
                    "meta_details": meta_details,
                }
            )

        if not candidate_scores:
            if verbose:
                print("\n[CACHE MISS]")
                print("Query:", query)
            result = {
                "hit": False,
                "score": float(best_score),
                "final_score": float(best_score),
                "hybrid_score": float(best_score),
                "metadata_score": 0.0,
                "metadata_decision": None,
                "response": None,
                "matched_query": best_entry["query"] if best_entry else None,
                "validator_used": False,
                "validator_confidence": None,
                "validator_safe_reuse": None,
                "validator_status": None,
                "dense_score": 0.0,
                "bm25_score": 0.0,
            }
            if best_entry:
                self._record_retrieval(best_entry, False, result)
            return result

        candidate_scores.sort(key=lambda item: item["final_score"], reverse=True)
        if debug:
            initial_preview = [
                (item["idx"], round(item["final_score"], 4))
                for item in candidate_scores[:5]
            ]
            log_debug(f"Initial rerank top: {initial_preview}", debug)

        top_candidates = candidate_scores[:VALIDATOR_TOP_K]
        if debug:
            top_ids = [(item["idx"], round(item["final_score"], 4)) for item in top_candidates]
            log_debug(f"Validator top candidates: {top_ids}", debug)

        validator_invoked = 0
        for candidate in top_candidates:
            hybrid_score = candidate["hybrid_score"]
            if self.validator is None:
                log_debug("Validator skipped: validator not configured", debug)
                continue
            if not (self.validator_min_hybrid <= hybrid_score <= self.validator_max_hybrid):
                log_debug(
                    "Validator skipped: score=%.4f outside range=(%.2f, %.2f)" %
                    (hybrid_score, self.validator_min_hybrid, self.validator_max_hybrid),
                    debug,
                )
                continue

            log_debug(
                "Validator activated: score=%.4f range=(%.2f, %.2f)" %
                (hybrid_score, self.validator_min_hybrid, self.validator_max_hybrid),
                debug,
            )
            validator_invoked += 1
            candidate["validator_used"] = True
            try:
                validator_result = self.validator.validate(
                    self.entries[candidate["idx"]]["query"],
                    query,
                    debug=debug,
                )
            except Exception as exc:
                validator_result = {
                    "safe_reuse": False,
                    "confidence": 0.0,
                    "status": "exception",
                    "error": str(exc),
                }

            validator_status = validator_result.get("status", "ok")
            candidate["validator_status"] = validator_status
            validator_technical_failure = validator_status in TECHNICAL_FAILURE_STATUSES

            if validator_status == "ok":
                safe_reuse = validator_result.get("safe_reuse")
                try:
                    validator_confidence = float(validator_result.get("confidence", 0.0))
                except (TypeError, ValueError):
                    validator_confidence = 0.0
                validator_confidence = max(0.0, min(1.0, validator_confidence))
                candidate["validator_confidence"] = float(validator_confidence)
                if isinstance(safe_reuse, bool):
                    candidate["validator_safe_reuse"] = safe_reuse
                if safe_reuse is False:
                    if validator_confidence >= VALIDATOR_HARD_REJECT_CONF:
                        log_warn(
                            "[VALIDATOR] Hard reject safe_reuse=false conf=%.2f" %
                            validator_confidence
                        )
                        candidate["hard_reject"] = True
                        continue
                    candidate["validator_bonus"] = (
                        -validator_confidence
                    ) * 0.5
                    log_debug(
                        "[VALIDATOR] Penalty=%.2f conf=%.2f" %
                        (candidate["validator_bonus"], validator_confidence),
                        debug,
                    )
                elif safe_reuse is True:
                    candidate["validator_bonus"] = (
                        validator_confidence - 0.5
                    ) * 0.5
                    log_debug(
                        "[VALIDATOR] Bonus=%.2f conf=%.2f" %
                        (candidate["validator_bonus"], validator_confidence),
                        debug,
                    )
            else:
                # Technical failures do not trigger a fail-open acceptance anymore.
                log_warn(f"[VALIDATOR] Technical failure status={validator_status}")
                if validator_technical_failure:
                    candidate["validator_bonus"] = -0.05
                    log_warn(
                        "[VALIDATOR] Technical failure penalty applied\n"
                        f"status={validator_status}\n"
                        "penalty=-0.05"
                    )

            candidate["final_score"] = (
                candidate["base_score"]
                + self.final_weight_validator * candidate["validator_bonus"]
            )

            if debug:
                validator_contrib = self.final_weight_validator * candidate["validator_bonus"]
                log_debug(
                    "Validated candidate idx=%s bonus=%.2f contrib=%.4f final=%.4f" %
                    (
                        candidate["idx"],
                        candidate["validator_bonus"],
                        validator_contrib,
                        candidate["final_score"],
                    ),
                    debug,
                )

        if debug:
            log_debug(
                "Validator invoked on %d/%d candidates" %
                (validator_invoked, len(top_candidates)),
                debug,
            )

        candidate_scores = [item for item in candidate_scores if not item.get("hard_reject")]
        if not candidate_scores:
            if verbose:
                print("\n[CACHE MISS]")
                print("Query:", query)
            result = {
                "hit": False,
                "score": float(best_score),
                "final_score": float(best_score),
                "hybrid_score": float(best_score),
                "metadata_score": 0.0,
                "metadata_decision": None,
                "response": None,
                "matched_query": best_entry["query"] if best_entry else None,
                "validator_used": False,
                "validator_confidence": None,
                "validator_safe_reuse": None,
                "validator_status": None,
                "dense_score": 0.0,
                "bm25_score": 0.0,
            }
            if best_entry:
                self._record_retrieval(best_entry, False, result)
            return result

        candidate_scores.sort(key=lambda item: item["final_score"], reverse=True)
        if debug:
            final_preview = [
                (item["idx"], round(item["final_score"], 4))
                for item in candidate_scores[:5]
            ]
            log_debug(f"Final rerank top: {final_preview}", debug)

        best = candidate_scores[0]
        best_entry = self.entries[best["idx"]]
        final_threshold, threshold_info = self._final_threshold(
            response_type,
            best["metadata_score"],
            best["validator_status"],
            best["hybrid_score"],
            best.get("validator_safe_reuse"),
            best.get("validator_confidence"),
        )

        if debug:
            top_preview = [
                (item["idx"], round(item["final_score"], 4))
                for item in candidate_scores[:5]
            ]
            log_debug(f"Rerank top: {top_preview}", debug)
            log_debug(
                "Best candidate idx=%s final=%.4f hybrid=%.4f metadata=%.2f validator=%.2f" %
                (
                    best["idx"],
                    best["final_score"],
                    best["hybrid_score"],
                    best["metadata_score"],
                    best["validator_confidence"],
                ),
                debug,
            )
            log_debug(
                "Threshold base=%.2f relax=%.2f final=%.2f" %
                (threshold_info["base"], threshold_info["relax"], threshold_info["final"]),
                debug,
            )
            log_debug(
                "Hybrid score evaluation: hybrid=%.4f final=%.4f" %
                (best["hybrid_score"], best["final_score"]),
                debug,
            )

        accept = best["final_score"] >= final_threshold
        # Fail-open fallback removed: do not accept on technical validator failures.

        if accept:
            log_info(
                "[ACCEPT] final_score=%.4f threshold=%.4f reason=final_score>=threshold" %
                (best["final_score"], final_threshold)
            )
        else:
            log_warn(
                "[REJECT] final_score=%.4f threshold=%.4f reason=below_threshold" %
                (best["final_score"], final_threshold)
            )

        if accept:
            if verbose:
                print("\n[CACHE HIT]")
                print("Query:", query)
                print("Matched:", best_entry["query"])
                print("Final Score:", round(float(best["final_score"]), 4))
                print("Hybrid:", round(float(best["hybrid_score"]), 4))
                print("Metadata:", round(float(best["metadata_score"]), 2))
                print("Validator:", round(float(best["validator_confidence"]), 2))
            result = {
                "hit": True,
                "score": float(best["final_score"]),
                "final_score": float(best["final_score"]),
                "hybrid_score": float(best["hybrid_score"]),
                "metadata_score": float(best["metadata_score"]),
                "metadata_decision": best["metadata_label"],
                "response": best_entry["response"],
                "matched_query": best_entry["query"],
                "validator_used": best["validator_used"],
                "validator_confidence": best["validator_confidence"],
                "validator_safe_reuse": best.get("validator_safe_reuse"),
                "validator_status": best["validator_status"],
                "dense_score": best["dense_score"],
                "bm25_score": best["bm25_score"],
            }
            self._record_retrieval(best_entry, True, result)
            return result

        if verbose:
            print("\n[CACHE MISS]")
            print("Query:", query)

        result = {
            "hit": False,
            "score": float(best["final_score"]),
            "final_score": float(best["final_score"]),
            "hybrid_score": float(best["hybrid_score"]),
            "metadata_score": float(best["metadata_score"]),
            "metadata_decision": best["metadata_label"],
            "response": None,
            "matched_query": best_entry["query"] if best_entry else None,
            "validator_used": best["validator_used"],
            "validator_confidence": best["validator_confidence"],
            "validator_safe_reuse": best.get("validator_safe_reuse"),
            "validator_status": best["validator_status"],
            "dense_score": best["dense_score"],
            "bm25_score": best["bm25_score"],
        }
        if best_entry:
            self._record_retrieval(best_entry, False, result)
        return result

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.entries = []
        self._bm25 = None
        self._bm25_corpus = []
        self._exact = {}

    @property
    def size(self) -> int:
        return self.index.ntotal


def download_csv(file_id: str, csv_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is not installed. Run: pip install gdown") from exc

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(csv_path), quiet=False)


def load_prompts(csv_path: Path):
    prompts = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompts.append((int(row["prompt_id"]), row["prompt"].strip()))
    return prompts


def run_simulation(prompts, verbose: bool, debug: bool, response_type: str) -> None:
    print("=" * 70)
    print("  PRODUCTION SIMULATION - semantic reuse")
    print("=" * 70)
    print()

    log_debug(f"Using Ollama host: {OLLAMA_HOST}", debug)
    log_debug(f"Using Ollama model: {OLLAMA_MODEL}", debug)
    log_debug(f"Response type hint: {response_type}", debug)

    validator = TinyLlamaValidator()
    cache = HybridSemanticCache(
        dim=DIM,
        top_k_dense=TOP_K_DENSE,
        top_k_bm25=TOP_K_BM25,
        default_threshold=DEFAULT_THRESHOLD,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        validator=validator,
        validator_min_hybrid=VALIDATOR_MIN_HYBRID,
        validator_max_hybrid=VALIDATOR_MAX_HYBRID,
    )
    hits_accepted = 0
    hits_rejected = 0
    misses = 0
    validator_approved = 0
    validator_rejected = 0
    validator_failed = 0
    reuse_correct = 0
    reuse_incorrect = 0
    reuse_unknown = 0
    hit_score_sum = 0.0
    hit_score_count = 0
    reject_score_sum = 0.0
    reject_score_count = 0
    candidate_score_sum = 0.0
    candidate_score_count = 0
    errors = []

    log_info("Pre-computing embeddings and metadata for all prompts...")
    embeddings_cache = {}
    metadata_cache = {}
    for pid, prompt in prompts:
        log_debug(f"Embedding+metadata for ID{pid:03d}", debug)
        embeddings_cache[pid] = embed(prompt)
        metadata_cache[pid] = extract_metadata(prompt)
    log_info(f"Done - {len(embeddings_cache)} embeddings cached in memory")

    for pid, prompt in prompts:
        emb = embeddings_cache[pid]
        meta = metadata_cache[pid]
        log_debug(f"Processing ID{pid:03d} q=\"{prompt[:60]}\"", debug)
        result = cache.search(
            prompt,
            embedding=emb,
            metadata=meta,
            response_type=response_type,
            verbose=verbose,
            debug=debug,
        )

        candidate_query = result.get("matched_query")
        candidate_exists = candidate_query is not None

        if result["hit"]:
            hits_accepted += 1
            hit_score_sum += result["final_score"]
            hit_score_count += 1
            status = "HIT  reuse accepted"
        elif candidate_exists:
            hits_rejected += 1
            reject_score_sum += result["final_score"]
            reject_score_count += 1
            status = "REJ  reuse rejected"
        else:
            misses += 1
            status = "MISS no candidate"

        if candidate_exists:
            candidate_score_sum += result["final_score"]
            candidate_score_count += 1

        eval_validation = None
        if candidate_exists:
            eval_validation = validator.validate(candidate_query, prompt, debug=debug)
            status_code = eval_validation.get("status")
            if status_code == "ok":
                if eval_validation.get("safe_reuse") is True:
                    validator_approved += 1
                else:
                    validator_rejected += 1
            else:
                validator_failed += 1

        if eval_validation and eval_validation.get("status") == "ok":
            safe_reuse = eval_validation.get("safe_reuse") is True
            if result["hit"] and safe_reuse:
                reuse_correct += 1
            elif result["hit"] and not safe_reuse:
                reuse_incorrect += 1
                errors.append(
                    f"  BAD [ID{pid:03d}] reuse accepted but validator rejected score={result['score']:.4f} "
                    f"q=\"{prompt[:50]}\""
                )
            elif (not result["hit"]) and safe_reuse:
                reuse_incorrect += 1
                errors.append(
                    f"  BAD [ID{pid:03d}] reuse rejected but validator approved score={result['score']:.4f} "
                    f"q=\"{prompt[:50]}\" candidate=\"{candidate_query[:50]}\""
                )
            else:
                reuse_correct += 1
        elif eval_validation is not None:
            reuse_unknown += 1

        print(f"  {status} [ID{pid:03d}] score={result['score']:.4f}")
        if verbose and eval_validation:
            conf = eval_validation.get("confidence")
            conf_value = float(conf) if isinstance(conf, (int, float)) else 0.0
            print(
                "      Validator status=%s safe_reuse=%s confidence=%.2f" %
                (
                    eval_validation.get("status"),
                    eval_validation.get("safe_reuse"),
                    conf_value,
                )
            )

        if not result["hit"]:
            cache.store(
                prompt,
                f"[LLM response for prompt {pid:03d}]",
                embedding=emb,
                metadata=meta,
            )

    total = len(prompts)
    reuse_known = reuse_correct + reuse_incorrect
    reuse_accuracy = (reuse_correct / reuse_known * 100) if reuse_known else 0
    avg_hit_score = (hit_score_sum / hit_score_count) if hit_score_count else 0.0
    avg_reject_score = (reject_score_sum / reject_score_count) if reject_score_count else 0.0
    avg_candidate_score = (
        candidate_score_sum / candidate_score_count
        if candidate_score_count
        else 0.0
    )

    print()
    print("=" * 70)
    print("  SEMANTIC CACHE - REUSE REPORT")
    print("=" * 70)
    print(f"  Response type hint     : {response_type}")
    print(f"  Total prompts          : {total}")
    print(f"  Cache hits accepted    : {hits_accepted}")
    print(f"  Cache hits rejected    : {hits_rejected}")
    print(f"  Cache misses           : {misses}")
    print("  ------------------------------")
    print(f"  Validator approved     : {validator_approved}")
    print(f"  Validator rejected     : {validator_rejected}")
    print(f"  Validator failures     : {validator_failed}")
    print("  ------------------------------")
    print(f"  Reuse correctness      : {reuse_accuracy:.1f}% (known={reuse_known})")
    print(f"  Reuse correct          : {reuse_correct}")
    print(f"  Reuse incorrect        : {reuse_incorrect}")
    print(f"  Reuse unknown          : {reuse_unknown}")
    print("  ------------------------------")
    print(f"  Avg final score (hit)  : {avg_hit_score:.4f}")
    print(f"  Avg final score (rej)  : {avg_reject_score:.4f}")
    print(f"  Avg final score (cand) : {avg_candidate_score:.4f}")
    print(f"  Cache size             : {cache.size}")
    print("=" * 70)

    if errors:
        print(f"\n  WARNING: {len(errors)} reuse mismatch(es):")
        for e in errors:
            print(e)


# ---------------------------------------------------------------------------
# Comparison experiment infrastructure
# ---------------------------------------------------------------------------

def run_experiment(
    prompts: list,
    cache_validator,
    embeddings_cache: dict,
    metadata_cache: dict,
    verbose: bool,
    debug: bool,
    response_type: str,
    label: str,
) -> list:
    """Run the incremental cache simulation with the given validator (or None).

    Returns a list of per-prompt result records for later metric computation.
    The cache is built incrementally (miss -> store) exactly as in run_simulation.
    """
    print()
    print("=" * 70)
    print(f"  RUNNING: {label}")
    print("=" * 70)
    print()

    cache = HybridSemanticCache(
        dim=DIM,
        top_k_dense=TOP_K_DENSE,
        top_k_bm25=TOP_K_BM25,
        default_threshold=DEFAULT_THRESHOLD,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        validator=cache_validator,
        validator_min_hybrid=VALIDATOR_MIN_HYBRID,
        validator_max_hybrid=VALIDATOR_MAX_HYBRID,
    )

    records = []
    for pid, prompt in prompts:
        emb = embeddings_cache[pid]
        meta = metadata_cache[pid]
        log_debug(f'Processing ID{pid:03d} q="{prompt[:60]}"', debug)
        result = cache.search(
            prompt,
            embedding=emb,
            metadata=meta,
            response_type=response_type,
            verbose=verbose,
            debug=debug,
        )

        hit = result["hit"]
        matched_query = result.get("matched_query")
        candidate_exists = matched_query is not None

        if hit:
            status = "HIT  reuse accepted"
        elif candidate_exists:
            status = "REJ  reuse rejected"
        else:
            status = "MISS no candidate"

        print(f"  {status} [ID{pid:03d}] score={result['score']:.4f}")

        # Determine threshold used for this result (re-derive from best candidate
        # data captured in the result dict; this does NOT change any scoring logic).
        _threshold_base = THRESHOLD_BY_TYPE.get(response_type, DEFAULT_THRESHOLD)
        _md_score_for_thresh = result.get("metadata_score", 0.0) or 0.0
        _val_status_for_thresh = result.get("validator_status") or "skipped"
        _hybrid_for_thresh = result.get("hybrid_score", 0.0) or 0.0
        _relax = 0.0
        if _md_score_for_thresh >= 0.85:
            _relax += THRESHOLD_RELAX_STRONG_METADATA
        elif _md_score_for_thresh >= 0.60:
            _relax += THRESHOLD_RELAX_MODERATE_METADATA
        if _val_status_for_thresh == "ok":
            _relax += THRESHOLD_RELAX_VALIDATOR_OK
        if _hybrid_for_thresh >= 0.90:
            _relax += THRESHOLD_RELAX_HIGH_HYBRID
        _relax = min(_relax, THRESHOLD_MAX_RELAX)
        _threshold_used = max(THRESHOLD_FLOOR, _threshold_base - _relax)

        records.append({
            "pid": pid,
            "prompt": prompt,
            "hit": hit,
            "final_score": result["final_score"],
            "hybrid_score": result.get("hybrid_score", 0.0),
            "metadata_score": result.get("metadata_score", 0.0),
            "metadata_decision": result.get("metadata_decision"),
            "matched_query": matched_query,
            "candidate_exists": candidate_exists,
            "validator_used": result.get("validator_used", False),
            "validator_status": result.get("validator_status"),
            "validator_confidence": result.get("validator_confidence"),
            "validator_safe_reuse": result.get("validator_safe_reuse"),
            "threshold": _threshold_used,
        })

        if not hit:
            cache.store(
                prompt,
                f"[LLM response for prompt {pid:03d}]",
                embedding=emb,
                metadata=meta,
            )

    return records


def compute_oracle_labels(
    all_records: list,
    oracle: TinyLlamaValidator,
    debug: bool,
) -> dict:
    """Run the oracle validator on every unique (candidate, prompt) pair.

    When TinyLlama returns a valid JSON response, that result is used.
    When TinyLlama fails (timeout / malformed JSON), a hybrid-score
    fallback is applied:

        hybrid >= ORACLE_FALLBACK_HIGH  ->  safe_reuse = True
        hybrid <  ORACLE_FALLBACK_LOW   ->  safe_reuse = False
        otherwise                       ->  None  (excluded from P/R/F1)

    The validator's internal memo avoids re-evaluating identical pairs.

    Returns
    -------
    dict mapping (candidate_query, prompt) -> True | False | None
    """
    ORACLE_FALLBACK_HIGH = 0.88   # very high similarity -> definitely safe to reuse
    ORACLE_FALLBACK_LOW  = 0.62   # below cache threshold -> definitely not safe

    # Build hybrid-score lookup (take the max across runs for any pair)
    hybrid_lookup: dict = {}
    for records in all_records:
        for rec in records:
            if rec["candidate_exists"] and rec["matched_query"]:
                key = (rec["matched_query"], rec["prompt"])
                hybrid_lookup[key] = max(
                    hybrid_lookup.get(key, 0.0),
                    rec.get("hybrid_score", 0.0),
                )

    pairs: set = set(hybrid_lookup.keys())

    print()
    log_info(f"Oracle pass: evaluating {len(pairs)} unique (candidate, prompt) pairs...")
    labels: dict = {}
    tinyllama_ok = 0
    fallback_used = 0
    fallback_unknown = 0

    for candidate, prompt in pairs:
        result = oracle.validate(candidate, prompt, debug=debug)
        status = result.get("status", "error")
        if status == "ok":
            labels[(candidate, prompt)] = result.get("safe_reuse")
            tinyllama_ok += 1
        else:
            # TinyLlama failed -> fall back to hybrid score
            hs = hybrid_lookup.get((candidate, prompt), 0.0)
            if hs >= ORACLE_FALLBACK_HIGH:
                labels[(candidate, prompt)] = True
                fallback_used += 1
            elif hs < ORACLE_FALLBACK_LOW:
                labels[(candidate, prompt)] = False
                fallback_used += 1
            else:
                labels[(candidate, prompt)] = None  # uncertain zone
                fallback_unknown += 1

    known = sum(1 for v in labels.values() if v is not None)
    log_info(
        f"Oracle complete: {known}/{len(labels)} pairs labelled "
        f"(TinyLlama={tinyllama_ok} fallback={fallback_used} uncertain={fallback_unknown})"
    )
    return labels


def compute_metrics(
    records: list,
    oracle_labels: dict,
    label: str,
) -> tuple:
    """Compute ExperimentMetrics from raw records and oracle ground-truth labels.

    Definitions
    -----------
    TP : cache accepted  AND oracle says safe_reuse=True
    FP : cache accepted  AND oracle says safe_reuse=False
    TN : cache rejected  AND oracle says safe_reuse=False
    FN : cache rejected  AND oracle says safe_reuse=True

    Prompts with no candidate (cold-start misses) and pairs where the oracle
    returned a technical failure are excluded from P/R/F1.

    Returns
    -------
    tuple of (ExperimentMetrics, fn_examples: list, fp_examples: list)
    """
    total = len(records)
    hits_accepted = sum(1 for r in records if r["hit"])
    hits_rejected = sum(1 for r in records if not r["hit"] and r["candidate_exists"])
    misses = sum(1 for r in records if not r["candidate_exists"])
    hit_rate = hits_accepted / total if total else 0.0

    accepted_scores = [r["final_score"] for r in records if r["hit"]]
    avg_score = sum(accepted_scores) / len(accepted_scores) if accepted_scores else 0.0

    tp = fp = tn = fn = 0
    reuse_accepted = reuse_rejected = 0
    fn_examples: list = []
    fp_examples: list = []

    for rec in records:
        if not rec["candidate_exists"] or not rec["matched_query"]:
            continue
        key = (rec["matched_query"], rec["prompt"])
        oracle_val = oracle_labels.get(key)
        if oracle_val is None:
            continue  # oracle technical failure -> exclude from P/R/F1

        hit = rec["hit"]
        if hit and oracle_val:
            tp += 1
            reuse_accepted += 1
        elif hit and not oracle_val:
            fp += 1
            fp_examples.append(rec)
        elif not hit and not oracle_val:
            tn += 1
            reuse_rejected += 1
        else:  # not hit and oracle_val is True
            fn += 1
            fn_examples.append(rec)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = ExperimentMetrics(
        label=label,
        total=total,
        hits_accepted=hits_accepted,
        hits_rejected=hits_rejected,
        misses=misses,
        hit_rate=hit_rate,
        avg_score=avg_score,
        reuse_accepted=reuse_accepted,
        reuse_rejected=reuse_rejected,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
    return metrics, fn_examples, fp_examples


def print_experiment_report(m: ExperimentMetrics) -> None:
    """Print a formatted single-run report block."""
    print()
    print("=" * 70)
    print(f"  {m.label}")
    print("=" * 70)
    print(f"  Total prompts          : {m.total}")
    print(f"  Cache hits             : {m.hits_accepted}")
    print(f"  Cache misses           : {m.misses + m.hits_rejected}")
    print(f"  Hit rate               : {m.hit_rate * 100:.1f}%")
    print(f"  Avg score (hits)       : {m.avg_score:.4f}")
    print("  ------------------------------")
    print(f"  Reuse accepted (TP)    : {m.tp}")
    print(f"  Reuse rejected (TN)    : {m.tn}")
    print("  ------------------------------")
    print(f"  Precision              : {m.precision:.4f}")
    print(f"  Recall                 : {m.recall:.4f}")
    print(f"  F1                     : {m.f1:.4f}")
    print("=" * 70)


def _fmt_val(v) -> str:
    """Format a value for analysis report output."""
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _diagnose_fn(rec: dict) -> str:
    """Derive a human-readable rejection reason for a false-negative record.

    Checks (in priority order):
    1. No candidate was retrieved at all (retrieval failure).
    2. Validator hard-rejected the pair.
    3. Metadata penalty dragged the final score below threshold.
    4. Hybrid score was below validator gate (never reached validator).
    5. Validator was not used at all (below gate range).
    6. Final score below threshold (generic).
    """
    if not rec.get("candidate_exists"):
        return "retrieval_failure: no candidate reached scoring"

    final = rec.get("final_score") or 0.0
    hybrid = rec.get("hybrid_score") or 0.0
    metadata = rec.get("metadata_score") or 0.0
    threshold = rec.get("threshold") or DEFAULT_THRESHOLD
    gap = threshold - final

    val_used = rec.get("validator_used", False)
    val_status = rec.get("validator_status") or "skipped"
    val_safe = rec.get("validator_safe_reuse")
    val_conf = rec.get("validator_confidence") or 0.0

    # Validator hard-rejected (safe_reuse=False, high confidence)
    if val_used and val_status == "ok" and val_safe is False and val_conf >= VALIDATOR_HARD_REJECT_CONF:
        return (
            f"validator_hard_reject: validator said safe_reuse=False "
            f"conf={val_conf:.2f} (>= hard-reject threshold {VALIDATOR_HARD_REJECT_CONF})"
        )

    # Validator penalty pushed score below threshold
    if val_used and val_status == "ok" and val_safe is False:
        return (
            f"validator_soft_penalty: validator said safe_reuse=False conf={val_conf:.2f}, "
            f"penalty dragged final={final:.4f} below threshold={threshold:.4f} (gap={gap:.4f})"
        )

    # Metadata score caused heavy penalty
    if metadata < 0.30 and gap > 0:
        return (
            f"metadata_penalty: low metadata_score={metadata:.4f} reduced final={final:.4f} "
            f"below threshold={threshold:.4f} (gap={gap:.4f})"
        )

    # Hybrid score was below the validator gate so validator never ran
    if not val_used and hybrid < VALIDATOR_MIN_HYBRID:
        return (
            f"below_validator_gate: hybrid={hybrid:.4f} < VALIDATOR_MIN_HYBRID={VALIDATOR_MIN_HYBRID}, "
            f"validator not invoked; final={final:.4f} below threshold={threshold:.4f} (gap={gap:.4f})"
        )

    # Validator was skipped (hybrid above max gate, or no validator configured)
    if not val_used and hybrid > VALIDATOR_MAX_HYBRID:
        return (
            f"above_validator_gate: hybrid={hybrid:.4f} > VALIDATOR_MAX_HYBRID={VALIDATOR_MAX_HYBRID}, "
            f"validator skipped; final={final:.4f} below threshold={threshold:.4f} (gap={gap:.4f})"
        )

    # Threshold simply too high relative to scores
    if gap > 0:
        return (
            f"threshold_too_high: final={final:.4f} below threshold={threshold:.4f} (gap={gap:.4f}), "
            f"hybrid={hybrid:.4f} metadata={metadata:.4f}"
        )

    return f"unknown: final={final:.4f} threshold={threshold:.4f} hybrid={hybrid:.4f}"


def _diagnose_fp(rec: dict) -> str:
    """Derive a human-readable acceptance reason for a false-positive record.

    Checks (in priority order):
    1. Validator approved the pair (safe_reuse=True).
    2. Validator was not invoked (hybrid below gate or no validator).
    3. Final score exceeded threshold without validator assistance.
    """
    final = rec.get("final_score") or 0.0
    hybrid = rec.get("hybrid_score") or 0.0
    metadata = rec.get("metadata_score") or 0.0
    threshold = rec.get("threshold") or DEFAULT_THRESHOLD
    margin = final - threshold

    val_used = rec.get("validator_used", False)
    val_status = rec.get("validator_status") or "skipped"
    val_safe = rec.get("validator_safe_reuse")
    val_conf = rec.get("validator_confidence") or 0.0

    if val_used and val_status == "ok" and val_safe is True:
        return (
            f"validator_approved: validator said safe_reuse=True conf={val_conf:.2f}, "
            f"bonus pushed final={final:.4f} above threshold={threshold:.4f} (margin={margin:.4f})"
        )

    if not val_used and hybrid < VALIDATOR_MIN_HYBRID:
        return (
            f"validator_not_invoked: hybrid={hybrid:.4f} < VALIDATOR_MIN_HYBRID={VALIDATOR_MIN_HYBRID}, "
            f"accepted on hybrid+metadata only; final={final:.4f} threshold={threshold:.4f} "
            f"(margin={margin:.4f})"
        )

    if not val_used and hybrid > VALIDATOR_MAX_HYBRID:
        return (
            f"high_hybrid_no_validator: hybrid={hybrid:.4f} > VALIDATOR_MAX_HYBRID={VALIDATOR_MAX_HYBRID}, "
            f"validator skipped; final={final:.4f} above threshold={threshold:.4f} (margin={margin:.4f})"
        )

    return (
        f"score_above_threshold: final={final:.4f} >= threshold={threshold:.4f} (margin={margin:.4f}), "
        f"hybrid={hybrid:.4f} metadata={metadata:.4f} validator_used={val_used}"
    )


def print_false_negative_report(fn_examples: list, label: str = "") -> None:
    """Print a detailed False Negative Analysis Report (first 50 examples).

    A false negative occurs when the cache rejected a reuse that the oracle
    considered safe (cache said MISS/REJECT but should have been a HIT).
    """
    header = f"FALSE NEGATIVE ANALYSIS" + (f" — {label}" if label else "")
    print()
    print("=" * 70)
    print(header)
    print("=" * 70)

    shown = fn_examples[:50]
    print(f"  Total false negatives : {len(fn_examples)}")
    print(f"  Showing first         : {len(shown)}")

    for i, rec in enumerate(shown, 1):
        print()
        print(f"  [{i}/{len(shown)}]")
        print(f"  Prompt               : {rec.get('prompt', 'N/A')}")
        print(f"  Matched Candidate    : {rec.get('matched_query', 'N/A')}")
        print(f"  Hybrid Score         : {_fmt_val(rec.get('hybrid_score'))}")
        print(f"  Metadata Score       : {_fmt_val(rec.get('metadata_score'))}")
        print(f"  Final Score          : {_fmt_val(rec.get('final_score'))}")
        print(f"  Threshold            : {_fmt_val(rec.get('threshold'))}")
        print(f"  Validator Used       : {_fmt_val(rec.get('validator_used'))}")
        print(f"  Validator Status     : {_fmt_val(rec.get('validator_status'))}")
        print(f"  Validator Confidence : {_fmt_val(rec.get('validator_confidence'))}")
        print(f"  Validator Safe Reuse : {_fmt_val(rec.get('validator_safe_reuse'))}")
        print(f"  Reason For Rejection : {_diagnose_fn(rec)}")
        print("  " + "-" * 60)

    print()
    print("=" * 70)


def print_false_positive_report(fp_examples: list, label: str = "") -> None:
    """Print a detailed False Positive Analysis Report (first 50 examples).

    A false positive occurs when the cache accepted a reuse that the oracle
    considered unsafe (cache said HIT but should have been a MISS/REJECT).
    """
    header = f"FALSE POSITIVE ANALYSIS" + (f" — {label}" if label else "")
    print()
    print("=" * 70)
    print(header)
    print("=" * 70)

    shown = fp_examples[:50]
    print(f"  Total false positives : {len(fp_examples)}")
    print(f"  Showing first         : {len(shown)}")

    for i, rec in enumerate(shown, 1):
        print()
        print(f"  [{i}/{len(shown)}]")
        print(f"  Prompt               : {rec.get('prompt', 'N/A')}")
        print(f"  Matched Candidate    : {rec.get('matched_query', 'N/A')}")
        print(f"  Hybrid Score         : {_fmt_val(rec.get('hybrid_score'))}")
        print(f"  Metadata Score       : {_fmt_val(rec.get('metadata_score'))}")
        print(f"  Final Score          : {_fmt_val(rec.get('final_score'))}")
        print(f"  Threshold            : {_fmt_val(rec.get('threshold'))}")
        print(f"  Validator Used       : {_fmt_val(rec.get('validator_used'))}")
        print(f"  Validator Status     : {_fmt_val(rec.get('validator_status'))}")
        print(f"  Validator Confidence : {_fmt_val(rec.get('validator_confidence'))}")
        print(f"  Validator Safe Reuse : {_fmt_val(rec.get('validator_safe_reuse'))}")
        print(f"  Reason For Acceptance: {_diagnose_fp(rec)}")
        print("  " + "-" * 60)

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Failure Analysis Summary
# ---------------------------------------------------------------------------

FN_CAT_BELOW_GATE   = "Below Validator Gate"
FN_CAT_THRESHOLD    = "Threshold Miss"
FN_CAT_METADATA     = "Metadata Penalty"
FN_CAT_VALIDATOR    = "Validator Rejection"
FN_CAT_NO_CANDIDATE = "No Candidate Retrieved"
FN_CAT_OTHER        = "Other"

FP_CAT_LOW_METADATA = "Accepted Despite Low Metadata"
FP_CAT_VAL_WARNING  = "Accepted Despite Validator Warning"
FP_CAT_NEAR_THRESH  = "Accepted Near Threshold"
FP_CAT_OTHER        = "Other"

# Near-threshold margin: FP within this margin above threshold
FP_NEAR_THRESHOLD_MARGIN = 0.03
# Low metadata threshold for FP categorization
FP_LOW_METADATA_SCORE = 0.50


def categorize_false_negative(rec: dict) -> str:
    """Assign a single category string to a false-negative record.

    Priority order matches _diagnose_fn() but maps to the summary categories
    displayed in the Failure Analysis Summary section.
    """
    if not rec.get("candidate_exists"):
        return FN_CAT_NO_CANDIDATE

    val_used   = rec.get("validator_used", False)
    val_status = rec.get("validator_status") or "skipped"
    val_safe   = rec.get("validator_safe_reuse")
    val_conf   = rec.get("validator_confidence") or 0.0
    hybrid     = rec.get("hybrid_score") or 0.0
    metadata   = rec.get("metadata_score") or 0.0
    final      = rec.get("final_score") or 0.0
    threshold  = rec.get("threshold") or DEFAULT_THRESHOLD
    gap        = threshold - final

    # Validator explicitly rejected the pair
    if val_used and val_status == "ok" and val_safe is False:
        return FN_CAT_VALIDATOR

    # Hybrid score was below the validator gate
    if not val_used and hybrid < VALIDATOR_MIN_HYBRID:
        return FN_CAT_BELOW_GATE

    # Metadata score was the primary drag (low score, non-trivial gap)
    if metadata < 0.30 and gap > 0:
        return FN_CAT_METADATA

    # Generic: final score fell below threshold
    if gap > 0:
        return FN_CAT_THRESHOLD

    return FN_CAT_OTHER


def categorize_false_positive(rec: dict) -> str:
    """Assign a single category string to a false-positive record."""
    val_used   = rec.get("validator_used", False)
    val_status = rec.get("validator_status") or "skipped"
    val_safe   = rec.get("validator_safe_reuse")
    metadata   = rec.get("metadata_score") or 0.0
    final      = rec.get("final_score") or 0.0
    threshold  = rec.get("threshold") or DEFAULT_THRESHOLD
    margin     = final - threshold

    # Validator signalled caution (safe_reuse=False) but the score still passed
    if val_used and val_status == "ok" and val_safe is False:
        return FP_CAT_VAL_WARNING

    # Metadata was low but score was high enough to pass
    if metadata < FP_LOW_METADATA_SCORE:
        return FP_CAT_LOW_METADATA

    # Accepted very close to the threshold (possible calibration issue)
    if 0 <= margin <= FP_NEAR_THRESHOLD_MARGIN:
        return FP_CAT_NEAR_THRESH

    return FP_CAT_OTHER


def print_failure_summary(
    fn_examples: list,
    fp_examples: list,
    label: str = "",
) -> None:
    """Print a Failure Analysis Summary for ALL FN/FP examples.

    Sections:
    - Categorized counts + percentages for every FN and FP.
    - Ranked top FN causes.
    - Automatic plain-English recommendation.
    """
    hdr = "FAILURE ANALYSIS SUMMARY" + (f" — {label}" if label else "")

    # ------------------------------------------------------------------ #
    # Categorize ALL false negatives
    # ------------------------------------------------------------------ #
    fn_cats: dict[str, int] = {
        FN_CAT_BELOW_GATE:   0,
        FN_CAT_THRESHOLD:    0,
        FN_CAT_METADATA:     0,
        FN_CAT_VALIDATOR:    0,
        FN_CAT_NO_CANDIDATE: 0,
        FN_CAT_OTHER:        0,
    }
    for rec in fn_examples:
        fn_cats[categorize_false_negative(rec)] += 1

    fn_total = len(fn_examples)

    # ------------------------------------------------------------------ #
    # Categorize ALL false positives
    # ------------------------------------------------------------------ #
    fp_cats: dict[str, int] = {
        FP_CAT_LOW_METADATA: 0,
        FP_CAT_VAL_WARNING:  0,
        FP_CAT_NEAR_THRESH:  0,
        FP_CAT_OTHER:        0,
    }
    for rec in fp_examples:
        fp_cats[categorize_false_positive(rec)] += 1

    fp_total = len(fp_examples)

    def _pct(n: int, total: int) -> str:
        if total == 0:
            return "  0.0%"
        return f"{n / total * 100:5.1f}%"

    # ------------------------------------------------------------------ #
    # Print
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print(hdr)
    print("=" * 70)

    # FALSE NEGATIVES
    print()
    print("  FALSE NEGATIVES (FN)")
    print("  " + "-" * 40)
    fn_rows = [
        (FN_CAT_BELOW_GATE,   "Below Validator Gate       "),
        (FN_CAT_THRESHOLD,    "Threshold Miss             "),
        (FN_CAT_METADATA,     "Metadata Penalty           "),
        (FN_CAT_VALIDATOR,    "Validator Rejection        "),
        (FN_CAT_NO_CANDIDATE, "No Candidate Retrieved     "),
        (FN_CAT_OTHER,        "Other                      "),
    ]
    for key, label_str in fn_rows:
        cnt = fn_cats[key]
        print(f"  {label_str}: {cnt:4d}  ({_pct(cnt, fn_total)})")
    print(f"  {'Total FN':<28}: {fn_total:4d}")

    # FALSE POSITIVES
    print()
    print("  FALSE POSITIVES (FP)")
    print("  " + "-" * 40)
    fp_rows = [
        (FP_CAT_LOW_METADATA, "Accepted Despite Low Metadata     "),
        (FP_CAT_VAL_WARNING,  "Accepted Despite Validator Warning"),
        (FP_CAT_NEAR_THRESH,  "Accepted Near Threshold           "),
        (FP_CAT_OTHER,        "Other                             "),
    ]
    for key, label_str in fp_rows:
        cnt = fp_cats[key]
        print(f"  {label_str}: {cnt:4d}  ({_pct(cnt, fp_total)})")
    print(f"  {'Total FP':<34}: {fp_total:4d}")

    # TOP FN CAUSES
    print()
    print("=" * 70)
    print("  TOP FN CAUSES")
    print("=" * 70)
    sorted_fn = sorted(fn_cats.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (cause, cnt) in enumerate(sorted_fn, 1):
        pct = cnt / fn_total * 100 if fn_total else 0.0
        print(f"  {rank}. {cause:<30}: {pct:5.1f}%  ({cnt} cases)")

    # RECOMMENDATION
    print()
    print("=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    if fn_total == 0:
        print("  No false negatives to analyse.")
    else:
        top_cause, top_cnt = sorted_fn[0]
        top_pct = top_cnt / fn_total * 100

        if top_pct > 50:
            if top_cause == FN_CAT_BELOW_GATE:
                print(
                    "  Most recall loss occurs before validation.\n"
                    f"  ({top_pct:.1f}% of FN are Below Validator Gate)\n"
                    "  Consider lowering VALIDATOR_MIN_HYBRID to allow the validator\n"
                    "  to evaluate more borderline candidates."
                )
            elif top_cause == FN_CAT_THRESHOLD:
                print(
                    "  Most recall loss occurs at the final threshold.\n"
                    f"  ({top_pct:.1f}% of FN are Threshold Miss)\n"
                    "  Consider threshold calibration or widening the\n"
                    "  THRESHOLD_RELAX_* relaxation parameters."
                )
            elif top_cause == FN_CAT_METADATA:
                print(
                    "  Metadata scoring is the main recall bottleneck.\n"
                    f"  ({top_pct:.1f}% of FN are Metadata Penalty)\n"
                    "  Review noun-overlap and entity-conflict thresholds in\n"
                    "  _metadata_score() to reduce over-penalisation."
                )
            elif top_cause == FN_CAT_VALIDATOR:
                print(
                    "  The validator is too conservative.\n"
                    f"  ({top_pct:.1f}% of FN are Validator Rejection)\n"
                    "  Consider raising VALIDATOR_HARD_REJECT_CONF or fine-tuning\n"
                    "  the validator prompt to reduce false rejections."
                )
            elif top_cause == FN_CAT_NO_CANDIDATE:
                print(
                    "  Most recall loss is due to retrieval failure.\n"
                    f"  ({top_pct:.1f}% of FN — no candidate reached scoring)\n"
                    "  Consider increasing TOP_K_DENSE / TOP_K_BM25 or improving\n"
                    "  the embedding coverage of the cache corpus."
                )
            else:
                print(
                    f"  Primary FN cause is '{top_cause}' ({top_pct:.1f}%).\n"
                    "  Inspect the detailed FN report above for further clues."
                )
        else:
            # No single dominant cause
            causes_str = ", ".join(
                f"{c} ({n / fn_total * 100:.1f}%)"
                for c, n in sorted_fn[:3]
                if n > 0
            )
            print(
                "  No single dominant FN cause (top causes: " + causes_str + ").\n"
                "  A multi-front investigation is recommended:\n"
                "   - Check hybrid score distribution for Below Validator Gate cases.\n"
                "   - Review threshold relaxation settings for Threshold Miss cases.\n"
                "   - Inspect metadata overlap for Metadata Penalty cases."
            )

    print("=" * 70)


def _delta_str(val: float, good_direction: int = 1) -> str:
    """Format a float delta with sign, directional arrow and a ✓/✗ annotation."""
    sign = "+" if val >= 0 else ""
    arrow = "\u25b2" if val > 0 else ("\u25bc" if val < 0 else "\u2500")
    is_good = (val > 0 and good_direction == 1) or (val < 0 and good_direction == -1)
    tag = " \u2713" if is_good else (" \u2717" if val != 0 else "")
    return f"{sign}{val:.4f} {arrow}{tag}"


def _int_delta_str(val: int, good_direction: int = 1) -> str:
    """Format an integer delta with sign, directional arrow and a ✓/✗ annotation."""
    sign = "+" if val >= 0 else ""
    arrow = "\u25b2" if val > 0 else ("\u25bc" if val < 0 else "\u2500")
    is_good = (val > 0 and good_direction == 1) or (val < 0 and good_direction == -1)
    tag = " \u2713" if is_good else (" \u2717" if val != 0 else "")
    return f"{sign}{val} {arrow}{tag}"


def print_comparison_table(
    baseline: ExperimentMetrics,
    validated: ExperimentMetrics,
) -> None:
    """Print the side-by-side validator impact comparison table."""
    hr_delta = validated.hit_rate - baseline.hit_rate
    p_delta = validated.precision - baseline.precision
    r_delta = validated.recall - baseline.recall
    f1_delta = validated.f1 - baseline.f1
    fp_reduced = baseline.fp - validated.fp
    fn_increased = validated.fn - baseline.fn
    net_benefit = fp_reduced - fn_increased

    fmt = "  {:<30} {:>10} {:>10} {:>18}"
    div = "  " + "-" * 70

    print()
    print("=" * 74)
    print("  COMPARISON: Validator Impact")
    print("=" * 74)
    print(fmt.format("Metric", "Baseline", "Validated", "Change"))
    print(div)
    print(fmt.format(
        "Hit rate",
        f"{baseline.hit_rate * 100:.1f}%",
        f"{validated.hit_rate * 100:.1f}%",
        _delta_str(hr_delta),
    ))
    print(fmt.format(
        "Precision",
        f"{baseline.precision:.4f}",
        f"{validated.precision:.4f}",
        _delta_str(p_delta),
    ))
    print(fmt.format(
        "Recall",
        f"{baseline.recall:.4f}",
        f"{validated.recall:.4f}",
        _delta_str(r_delta),
    ))
    print(fmt.format(
        "F1",
        f"{baseline.f1:.4f}",
        f"{validated.f1:.4f}",
        _delta_str(f1_delta),
    ))
    print(div)
    print(fmt.format(
        "TP (correct accepts)",
        str(baseline.tp), str(validated.tp),
        _int_delta_str(validated.tp - baseline.tp),
    ))
    print(fmt.format(
        "FP (bad accepts)",
        str(baseline.fp), str(validated.fp),
        _int_delta_str(validated.fp - baseline.fp, good_direction=-1),
    ))
    print(fmt.format(
        "TN (correct rejects)",
        str(baseline.tn), str(validated.tn),
        _int_delta_str(validated.tn - baseline.tn),
    ))
    print(fmt.format(
        "FN (missed reuses)",
        str(baseline.fn), str(validated.fn),
        _int_delta_str(validated.fn - baseline.fn, good_direction=-1),
    ))
    print(div)
    print(f"  {'False positives reduced':<30} {fp_reduced:>+10d}")
    print(f"  {'False negatives increased':<30} {fn_increased:>+10d}")
    if net_benefit > 0:
        verdict = "\u2713 Validator helps"
    elif net_benefit < 0:
        verdict = "\u2717 Validator hurts"
    else:
        verdict = "\u2500 No net change"
    print(f"  {'Net benefit (FP reduced - FN increased)':<30} {net_benefit:>+10d}  {verdict}")
    print("=" * 74)


def run_compare_validator(
    prompts: list,
    verbose: bool,
    debug: bool,
    response_type: str,
) -> None:
    """Run baseline vs. TinyLlama validator comparison and print side-by-side report.

    Guarantees
    ----------
    - Identical prompt sequence for both runs.
    - Identical embeddings, BM25 corpus, and metadata (pre-computed once).
    - Identical cache configuration and retrieval logic.
    - The ONLY difference between runs is cache_validator=None vs. TinyLlamaValidator().
    - A single oracle TinyLlamaValidator evaluates ground-truth for every
      (candidate, prompt) pair seen in either run.  Its internal memo ensures
      each unique pair is evaluated at most once.
    """
    print()
    print("=" * 70)
    print("  VALIDATOR COMPARISON EXPERIMENT")
    print("=" * 70)
    print()

    # Pre-compute embeddings and metadata once, shared across both runs.
    log_info("Pre-computing embeddings and metadata (shared across both runs)...")
    embeddings_cache: dict = {}
    metadata_cache: dict = {}
    for pid, prompt in prompts:
        log_debug(f"Embedding+metadata for ID{pid:03d}", debug)
        embeddings_cache[pid] = embed(prompt)
        metadata_cache[pid] = extract_metadata(prompt)
    log_info(f"Done - {len(embeddings_cache)} embeddings ready")

    # --- Pass 1: Baseline (validator disabled) ---
    log_info("Pass 1 of 2: Baseline (no validator)")
    baseline_records = run_experiment(
        prompts=prompts,
        cache_validator=None,
        embeddings_cache=embeddings_cache,
        metadata_cache=metadata_cache,
        verbose=verbose,
        debug=debug,
        response_type=response_type,
        label="BASELINE CACHE (NO VALIDATOR)",
    )

    # --- Pass 2: Cache + TinyLlama Validator ---
    log_info("Pass 2 of 2: Cache + TinyLlama Validator")
    validated_validator = TinyLlamaValidator()
    validated_records = run_experiment(
        prompts=prompts,
        cache_validator=validated_validator,
        embeddings_cache=embeddings_cache,
        metadata_cache=metadata_cache,
        verbose=verbose,
        debug=debug,
        response_type=response_type,
        label="CACHE + TINYLLAMA VALIDATOR",
    )

    # --- Oracle pass ---
    # Reuse validated_validator so its memo (already warm from pass 2) avoids
    # redundant LLM calls for pairs that were already evaluated by the cache.
    oracle_labels = compute_oracle_labels(
        all_records=[baseline_records, validated_records],
        oracle=validated_validator,
        debug=debug,
    )

    # --- Compute metrics for each run ---
    baseline_metrics, baseline_fn, baseline_fp = compute_metrics(
        baseline_records, oracle_labels, "BASELINE CACHE (NO VALIDATOR)"
    )
    validated_metrics, validated_fn, validated_fp = compute_metrics(
        validated_records, oracle_labels, "CACHE + TINYLLAMA VALIDATOR"
    )

    # --- Print reports, disagreements, and comparison table ---
    print_experiment_report(baseline_metrics)
    print_experiment_report(validated_metrics)
    print_disagreements(baseline_records, validated_records)
    print_comparison_table(baseline_metrics, validated_metrics)

    # --- False Negative / False Positive Analysis ---
    print_false_negative_report(baseline_fn, label="BASELINE (NO VALIDATOR)")
    print_false_positive_report(baseline_fp, label="BASELINE (NO VALIDATOR)")
    print_false_negative_report(validated_fn, label="CACHE + TINYLLAMA VALIDATOR")
    print_false_positive_report(validated_fp, label="CACHE + TINYLLAMA VALIDATOR")

    # --- Failure Analysis Summary + Recommendation ---
    print_failure_summary(baseline_fn, baseline_fp, label="BASELINE (NO VALIDATOR)")
    print_failure_summary(validated_fn, validated_fp, label="CACHE + TINYLLAMA VALIDATOR")


def print_disagreements(
    baseline_records: list,
    validated_records: list,
) -> None:
    """Print every prompt where baseline and validated runs made opposite decisions.

    WITHOUT VALIDATOR: HIT  / WITH VALIDATOR: REJECT  -> validator blocked a reuse
    WITHOUT VALIDATOR: REJECT / WITH VALIDATOR: HIT   -> validator enabled a reuse
    """
    baseline_by_pid = {r["pid"]: r for r in baseline_records}
    validated_by_pid = {r["pid"]: r for r in validated_records}

    disagreements = []
    for pid in sorted(set(baseline_by_pid) | set(validated_by_pid)):
        base = baseline_by_pid.get(pid)
        val  = validated_by_pid.get(pid)
        if base is None or val is None:
            continue
        if base["hit"] != val["hit"]:
            disagreements.append((base, val))

    print()
    print("=" * 70)
    print(f"  DISAGREEMENTS  ({len(disagreements)} case{'s' if len(disagreements) != 1 else ''})")
    print("=" * 70)

    if not disagreements:
        print("  (none — both runs agreed on every prompt)")
        print("=" * 70)
        return

    for base, val in disagreements:
        print()
        if base["hit"] and not val["hit"]:
            print("  WITHOUT VALIDATOR: HIT")
            print("  WITH VALIDATOR:    REJECT")
            cached     = base["matched_query"] or "(none)"
            base_score = base["final_score"]
            val_score  = val["final_score"]
        else:
            print("  WITHOUT VALIDATOR: REJECT")
            print("  WITH VALIDATOR:    HIT")
            cached     = val["matched_query"] or base["matched_query"] or "(none)"
            base_score = base["final_score"]
            val_score  = val["final_score"]

        print()
        print(f"  Query:                {base['prompt']}")
        print(f"  Cached:               {cached}")
        print(f"  Score (no validator): {base_score:.4f}")
        print(f"  Score (validated):    {val_score:.4f}")
        print("  ----------------------------------")

    print("=" * 70)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Hybrid semantic cache demo")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Path to test.csv")
    parser.add_argument("--no-download", action="store_true", help="Do not download CSV if missing")
    parser.add_argument("--file-id", default=FILE_ID, help="Google Drive file id for test.csv")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Verbose cache/validator output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--log-csv", action="store_true", help="Write terminal output to a CSV log")
    parser.add_argument("--log-csv-path", default=str(Path(__file__).with_name("output.csv")), help="Path to terminal CSV log")
    parser.add_argument(
        "--response-type",
        default="general",
        choices=sorted(THRESHOLD_BY_TYPE.keys()),
        help="Response type hint for thresholding",
    )
    parser.add_argument(
        "--compare-validator",
        action="store_true",
        help="Run baseline vs. TinyLlama validator comparison experiment",
    )
    return parser.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)

    csv_log_path = Path(args.log_csv_path)
    if not getattr(args, "log_csv", False):
        csv_log_path = Path(__file__).with_name("output.csv")

    try:
        enable_csv_output(csv_log_path)
        log_info(f"CSV terminal logging enabled -> {csv_log_path}")
    except Exception as exc:
        log_warn(f"Failed to enable CSV logging: {exc}")

    if args.debug:
        log_debug(
            "Config: model=%s dim=%s top_k_dense=%s top_k_bm25=%s" %
            (MODEL_NAME, DIM, TOP_K_DENSE, TOP_K_BM25),
            True,
        )
        log_debug(
            "Weights: dense=%.2f bm25=%.2f" %
            (DENSE_WEIGHT, BM25_WEIGHT),
            True,
        )
        log_debug(
            "Final weights: hybrid=%.2f metadata=%.2f validator=%.2f" %
            (FINAL_WEIGHT_HYBRID, FINAL_WEIGHT_METADATA, FINAL_WEIGHT_VALIDATOR),
            True,
        )
        log_debug(f"Thresholds: {THRESHOLD_BY_TYPE}", True)
        log_debug(
            "Validator gate: min=%.2f max=%.2f min_conf=%.2f" %
            (VALIDATOR_MIN_HYBRID, VALIDATOR_MAX_HYBRID, VALIDATOR_MIN_CONFIDENCE),
            True,
        )

    csv_path = Path(args.csv)
    if not csv_path.exists() and not args.no_download:
        log_info(f"test.csv not found at {csv_path}, downloading...")
        download_csv(args.file_id, csv_path)

    if not csv_path.exists():
        log_error(f"CSV not found: {csv_path}")
        return 1

    load_models(debug=args.debug)

    prompts = load_prompts(csv_path)
    log_info(f"Loaded {len(prompts)} prompts from {csv_path}")
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]
        log_info(f"Limiting prompts to first {len(prompts)}")

    if getattr(args, "compare_validator", False):
        run_compare_validator(
            prompts,
            verbose=args.verbose,
            debug=args.debug,
            response_type=args.response_type,
        )
    else:
        run_simulation(prompts, verbose=args.verbose, debug=args.debug, response_type=args.response_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
