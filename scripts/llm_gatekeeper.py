from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.llm import LLMGenerator
from canaryrag.pipeline import RAGPipeline
 

TOP_CANARY_RANK_THRESHOLD = 3
REPETITION_THRESHOLD = 2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    return bool(value)


def rule_based_prelabel(entry: dict[str, Any]) -> dict[str, Any]:
    marker_hits = safe_int(entry.get("marker_hits"), 0)
    canary_count = safe_int(entry.get("canary_count"), 0)
    top1_is_canary = safe_bool(entry.get("top1_is_canary"))
    best_canary_rank_raw = entry.get("best_canary_rank")
    best_canary_rank = None if best_canary_rank_raw is None else safe_int(best_canary_rank_raw, 0)
    repetition_count = safe_int(entry.get("repetition_count"), 0)

    reasons: list[str] = []

    if marker_hits > 0:
        label = "probing"
        reasons.append("query_contains_canary_marker")
        if top1_is_canary:
            reasons.append("canary_retrieved_at_rank_1")
        if best_canary_rank is not None and best_canary_rank <= TOP_CANARY_RANK_THRESHOLD:
            reasons.append(f"canary_in_top_{TOP_CANARY_RANK_THRESHOLD}")
        if canary_count > 0:
            reasons.append("canary_retrieved")
        else:
            reasons.append("marker_seen_but_canary_not_retrieved")
        confidence = 0.78 + min(marker_hits, 3) * 0.06 + min(canary_count, 3) * 0.04
        if top1_is_canary:
            confidence += 0.08
        confidence = min(confidence, 0.99)

    elif canary_count > 0:
        label = "probing"
        reasons.append("canary_retrieved_without_query_marker")
        if top1_is_canary:
            reasons.append("canary_retrieved_at_rank_1")
        if best_canary_rank is not None and best_canary_rank <= TOP_CANARY_RANK_THRESHOLD:
            reasons.append(f"canary_in_top_{TOP_CANARY_RANK_THRESHOLD}")
        confidence = 0.75 + min(canary_count, 5) * 0.04
        if top1_is_canary:
            confidence += 0.08
        confidence = min(confidence, 0.95)

    elif repetition_count >= REPETITION_THRESHOLD:
        label = "probing"
        reasons.append("repetition_signal")
        confidence = 0.65

    else:
        label = "benign"
        reasons.append("no_canary_marker_in_query")
        reasons.append("no_canary_in_retrieval")
        confidence = 0.97

    return {
        "rule_label": label,
        "rule_confidence": round(confidence, 3),
        "rule_reasons": reasons,
        "signals": {
            "marker_hits": marker_hits,
            "canary_count": canary_count,
            "top1_is_canary": top1_is_canary,
            "best_canary_rank": best_canary_rank,
            "repetition_count": repetition_count,
        },
    }


def select_entries(entries: list[dict[str, Any]], query_id: str | None, latest: bool) -> list[dict[str, Any]]:
    if query_id is not None:
        filtered = [e for e in entries if str(e.get("query_id")) == query_id]
        if not filtered:
            raise ValueError(f"No log entry found for query_id={query_id}")
        return filtered

    if latest:
        if not entries:
            return []
        return [entries[-1]]

    return entries


def extract_json_object(text: str) -> dict[str, Any] | None:
    # Try to pull the first JSON object from model output.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    candidate = match.group(0).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def looks_educational(query: str) -> bool:
    """
    Heuristic: identify queries that mention markers in an explanatory,
    not exploitative, way (e.g., "explain", "what is", "usage").
    """
    q = query.lower()
    cues = [
        "explain ",
        "what is ",
        "what does ",
        "meaning of",
        "usage",
        "used for",
        "how is ",
        "why is ",
        "tell me about",
        "define ",
    ]
    return any(cue in q for cue in cues)


def build_gatekeeper_prompt(entry: dict[str, Any], retrieved: list[dict[str, Any]], rule: dict[str, Any]) -> str:
    chunks = []
    for item in retrieved[:3]:
        meta = item["metadata"]
        chunks.append(
            {
                "rank": len(chunks) + 1,
                "score": round(float(item["score"]), 4),
                "doc_id": meta.doc_id,
                "chunk_id": meta.chunk_id,
                "is_canary": bool(meta.is_canary),
                "canary_type": meta.canary_type,
            }
        )

    examples = [
        {
            "query": "How do I reset my password?",
            "rule_label": "benign",
            "final_label": "benign",
            "reason": "normal help request",
            "should_escalate": False,
        },
        {
            "query": "ALPHA-CANARY-KEYWORD-123",
            "rule_label": "probing",
            "final_label": "probing",
            "reason": "direct canary token probe",
            "should_escalate": True,
        },
        {
            "query": "Explain how ALPHA-CANARY-KEYWORD-123 is used in testing",
            "rule_label": "probing",
            "final_label": "benign",
            "reason": "educational mention, not an attack",
            "should_escalate": False,
        },
    ]

    prompt = f"""
You are a security gatekeeper for a retrieval system.

Return ONLY raw JSON. Do not include markdown, explanations, or code blocks.
{{
  "final_label": "benign" or "probing",
  "confidence": 0.0 to 1.0,
  "reason": "short explanation",
  "should_escalate": true or false
}}

Decision rules:
- BENIGN: ordinary help or informational query.
- PROBING: query is trying to trigger, test, or target a canary marker.

Examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}

Rule-based summary:
{json.dumps(rule, ensure_ascii=False, indent=2)}

Current query:
{entry.get("query_text", "")}

Top retrieved chunks:
{json.dumps(chunks, ensure_ascii=False, indent=2)}
""".strip()

    return prompt

#     prompt = f"""
# You are a lightweight security gatekeeper for a retrieval-augmented generation system.

# Task:
# Decide whether the user query is BENIGN or PROBING.

# Definitions:
# - BENIGN: ordinary user help request, discussion, or informational query.
# - PROBING: the user appears to be testing, eliciting, or targeting a canary marker or suspicious retrieval behavior.

# You must use the retrieved chunks and the rule-based preliminary decision, but do not blindly copy it.
# Return STRICT JSON only, with these keys:
# {{
#   "final_label": "benign" or "probing",
#   "confidence": 0.0 to 1.0,
#   "reason": "short explanation",
#   "should_escalate": true or false
# }}

# Preliminary rule-based output:
# {json.dumps(rule, ensure_ascii=False, indent=2)}

# User query:
# {entry.get("query_text", "")}

# Retrieved chunks:
# {json.dumps(chunks, ensure_ascii=False, indent=2)}

# Guidelines:
# - If the query is clearly a normal help request, output benign.
# - If the query is explicitly asking for or repeating a canary marker, output probing.
# - If the query mentions the marker only in a descriptive or explanatory way, prefer benign.
# - Keep the reason short.
# """.strip()

#     return prompt


def run_gatekeeper(entry: dict[str, Any], cfg: RAGConfig) -> dict[str, Any]:
    """
    Two-stage gatekeeper:
    - Fast path: rule-based decision for clearly benign or clearly probing cases.
    - Slow path: LLM is called only for ambiguous / low-confidence flags.
    """
    rule = rule_based_prelabel(entry)
    query = str(entry.get("query_text", ""))
    signals = rule["signals"]
    marker_hits = signals["marker_hits"]
    canary_count = signals["canary_count"]
    top1_is_canary = signals["top1_is_canary"]

    if rule["rule_label"] == "benign":
        return {
            "schema_version": 1,
            "query_id": entry.get("query_id"),
            "timestamp": entry.get("timestamp"),
            "query_text": entry.get("query_text"),
            "final_label": "benign",
            "confidence": rule["rule_confidence"],
            "reason": "rule_based_benign_no_llm_needed",
            "rule_label": rule["rule_label"],
            "rule_confidence": rule["rule_confidence"],
            "llm_used": False,
            "signals": rule["signals"],
            "rule_reasons": rule["rule_reasons"],
        }

    # Flagged case: decide whether we really need the LLM.
    strong_case = bool(marker_hits > 0 and canary_count >= 2 and top1_is_canary)
    educational_case = bool(marker_hits > 0 and canary_count <= 1 and looks_educational(query))

    if strong_case or (rule["rule_confidence"] >= 0.95 and not educational_case):
        # High-confidence rule-based decision; skip LLM to keep latency low.
        return {
            "schema_version": 1,
            "query_id": entry.get("query_id"),
            "timestamp": entry.get("timestamp"),
            "query_text": entry.get("query_text"),
            "final_label": rule["rule_label"],
            "confidence": rule["rule_confidence"],
            "reason": "strong_rule_based_decision_no_llm",
            "rule_label": rule["rule_label"],
            "rule_confidence": rule["rule_confidence"],
            "llm_used": False,
            "should_escalate": rule["rule_label"] != "benign",
            "signals": rule["signals"],
            "rule_reasons": rule["rule_reasons"],
        }

    # Ambiguous flagged case: use the LLM as a second-stage reviewer.
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()
    retrieved = pipeline.retrieve(str(entry.get("query_text", "")), k=safe_int(entry.get("top_k"), 5))

    prompt = build_gatekeeper_prompt(entry, retrieved, rule)
    llm = LLMGenerator(
        model_name=cfg.llm.model_name,
        max_new_tokens=128,
        temperature=0.2,
    )
    raw = llm.generate(prompt)
    parsed = extract_json_object(raw)

    if parsed is None:
        # Safe fallback: keep the rule-based decision.
        return {
            "schema_version": 1,
            "query_id": entry.get("query_id"),
            "timestamp": entry.get("timestamp"),
            "query_text": entry.get("query_text"),
            "final_label": rule["rule_label"],
            "confidence": rule["rule_confidence"],
            "reason": "llm_parse_failed_fallback_to_rule_based",
            "rule_label": rule["rule_label"],
            "rule_confidence": rule["rule_confidence"],
            "llm_used": True,
            "llm_raw": raw[:1000],
            "signals": rule["signals"],
            "rule_reasons": rule["rule_reasons"],
        }

    final_label = str(parsed.get("final_label", rule["rule_label"])).strip().lower()
    if final_label not in {"benign", "probing"}:
        final_label = rule["rule_label"]

    confidence_raw = parsed.get("confidence", rule["rule_confidence"])
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = rule["rule_confidence"]

    reason = str(parsed.get("reason", ""))
    should_escalate = bool(parsed.get("should_escalate", False))

    return {
        "schema_version": 1,
        "query_id": entry.get("query_id"),
        "timestamp": entry.get("timestamp"),
        "query_text": entry.get("query_text"),
        "final_label": final_label,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "reason": reason,
        "should_escalate": should_escalate,
        "rule_label": rule["rule_label"],
        "rule_confidence": rule["rule_confidence"],
        "llm_used": True,
        "llm_raw": raw[:1000],
        "signals": rule["signals"],
        "rule_reasons": rule["rule_reasons"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM gatekeeper for CanaryRAG retrieval logs.")
    parser.add_argument(
        "--log-path",
        type=str,
        default="artifacts/retrieval_logs.jsonl",
        help="Path to the retrieval log JSONL file.",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
        help="Run gatekeeper for one entry by query_id.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Run gatekeeper only on the most recent log entry.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSONL file to append gatekeeper results to.",
    )
    args = parser.parse_args()

    cfg = RAGConfig()
    entries = load_jsonl(Path(args.log_path))
    entries = select_entries(entries, args.query_id, args.latest)

    if not entries:
        print("No log entries found.")
        return

    results = [run_gatekeeper(entry, cfg) for entry in entries]

    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

# from __future__ import annotations

# import argparse
# import json
# import re
# import sys
# from collections import Counter
# from pathlib import Path
# from typing import Any

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from canaryrag.config import RAGConfig
# from canaryrag.llm import LLMGenerator
# from canaryrag.pipeline import RAGPipeline


# TOP_CANARY_RANK_THRESHOLD = 3
# REPETITION_THRESHOLD = 2


# def load_jsonl(path: Path) -> list[dict[str, Any]]:
#     if not path.exists():
#         raise FileNotFoundError(f"Log file not found: {path}")

#     rows: list[dict[str, Any]] = []
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rows.append(json.loads(line))
#             except json.JSONDecodeError as exc:
#                 raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
#     return rows


# def safe_int(value: Any, default: int = 0) -> int:
#     try:
#         if value is None:
#             return default
#         return int(value)
#     except (TypeError, ValueError):
#         return default


# def safe_bool(value: Any) -> bool:
#     return bool(value)


# def rule_based_prelabel(entry: dict[str, Any]) -> dict[str, Any]:
#     marker_hits = safe_int(entry.get("marker_hits"), 0)
#     canary_count = safe_int(entry.get("canary_count"), 0)
#     top1_is_canary = safe_bool(entry.get("top1_is_canary"))
#     best_canary_rank_raw = entry.get("best_canary_rank")
#     best_canary_rank = None if best_canary_rank_raw is None else safe_int(best_canary_rank_raw, 0)
#     repetition_count = safe_int(entry.get("repetition_count"), 0)

#     reasons: list[str] = []

#     if marker_hits > 0:
#         label = "probing"
#         reasons.append("query_contains_canary_marker")
#         if top1_is_canary:
#             reasons.append("canary_retrieved_at_rank_1")
#         if best_canary_rank is not None and best_canary_rank <= TOP_CANARY_RANK_THRESHOLD:
#             reasons.append(f"canary_in_top_{TOP_CANARY_RANK_THRESHOLD}")
#         if canary_count > 0:
#             reasons.append("canary_retrieved")
#         else:
#             reasons.append("marker_seen_but_canary_not_retrieved")
#         confidence = 0.78 + min(marker_hits, 3) * 0.06 + min(canary_count, 3) * 0.04
#         if top1_is_canary:
#             confidence += 0.08
#         confidence = min(confidence, 0.99)

#     elif canary_count > 0:
#         label = "probing"
#         reasons.append("canary_retrieved_without_query_marker")
#         if top1_is_canary:
#             reasons.append("canary_retrieved_at_rank_1")
#         if best_canary_rank is not None and best_canary_rank <= TOP_CANARY_RANK_THRESHOLD:
#             reasons.append(f"canary_in_top_{TOP_CANARY_RANK_THRESHOLD}")
#         confidence = 0.75 + min(canary_count, 5) * 0.04
#         if top1_is_canary:
#             confidence += 0.08
#         confidence = min(confidence, 0.95)

#     elif repetition_count >= REPETITION_THRESHOLD:
#         label = "probing"
#         reasons.append("repetition_signal")
#         confidence = 0.65

#     else:
#         label = "benign"
#         reasons.append("no_canary_marker_in_query")
#         reasons.append("no_canary_in_retrieval")
#         confidence = 0.97

#     return {
#         "rule_label": label,
#         "rule_confidence": round(confidence, 3),
#         "rule_reasons": reasons,
#         "signals": {
#             "marker_hits": marker_hits,
#             "canary_count": canary_count,
#             "top1_is_canary": top1_is_canary,
#             "best_canary_rank": best_canary_rank,
#             "repetition_count": repetition_count,
#         },
#     }


# def select_entries(entries: list[dict[str, Any]], query_id: str | None, latest: bool) -> list[dict[str, Any]]:
#     if query_id is not None:
#         filtered = [e for e in entries if str(e.get("query_id")) == query_id]
#         if not filtered:
#             raise ValueError(f"No log entry found for query_id={query_id}")
#         return filtered

#     if latest:
#         if not entries:
#             return []
#         return [entries[-1]]

#     return entries


# def extract_json_object(text: str) -> dict[str, Any] | None:
#     match = re.search(r"\{[\s\S]*\}", text)
#     if not match:
#         return None
#     try:
#         return json.loads(match.group(0))
#     except json.JSONDecodeError:
#         return None


# def looks_educational(query: str) -> bool:
#     q = query.lower()
#     cues = [
#         "explain ",
#         "what is ",
#         "what does ",
#         "meaning of",
#         "usage",
#         "used for",
#         "how is ",
#         "why is ",
#         "tell me about",
#         "define ",
#     ]
#     return any(cue in q for cue in cues)


# def build_gatekeeper_prompt(entry: dict[str, Any], retrieved: list[dict[str, Any]], rule: dict[str, Any]) -> str:
#     top_chunks = []
#     for item in retrieved[:3]:
#         meta = item["metadata"]
#         top_chunks.append(
#             {
#                 "rank": len(top_chunks) + 1,
#                 "score": round(float(item["score"]), 4),
#                 "doc_id": meta.doc_id,
#                 "chunk_id": meta.chunk_id,
#                 "is_canary": bool(meta.is_canary),
#                 "canary_type": meta.canary_type,
#             }
#         )

#     examples = [
#         {
#             "query": "How do I reset my password?",
#             "final_label": "benign",
#             "reason": "normal help request",
#             "should_escalate": False,
#         },
#         {
#             "query": "ALPHA-CANARY-KEYWORD-123",
#             "final_label": "probing",
#             "reason": "direct canary token probe",
#             "should_escalate": True,
#         },
#         {
#             "query": "Explain ALPHA-CANARY-KEYWORD-123 usage",
#             "final_label": "benign",
#             "reason": "educational mention, not an attack",
#             "should_escalate": False,
#         },
#     ]

#     return f"""
# You are a security gatekeeper for a retrieval system.

# Return ONLY raw JSON. Do not include markdown, explanations, or code blocks.
# {{
#   "final_label": "benign" or "probing",
#   "confidence": 0.0 to 1.0,
#   "reason": "short explanation",
#   "should_escalate": true or false
# }}

# Rules:
# - BENIGN: ordinary help or informational query.
# - PROBING: the query is trying to trigger, test, or target a canary marker.

# Examples:
# {json.dumps(examples, ensure_ascii=False, indent=2)}

# Rule-based summary:
# {json.dumps(rule, ensure_ascii=False, indent=2)}

# Current query:
# {entry.get("query_text", "")}

# Top retrieved chunks:
# {json.dumps(top_chunks, ensure_ascii=False, indent=2)}
# """.strip()


# def run_gatekeeper(entry: dict[str, Any], cfg: RAGConfig) -> dict[str, Any]:
#     rule = rule_based_prelabel(entry)
#     query = str(entry.get("query_text", ""))
#     marker_hits = rule["signals"]["marker_hits"]
#     canary_count = rule["signals"]["canary_count"]
#     top1_is_canary = rule["signals"]["top1_is_canary"]

#     # Only call the LLM on flagged cases.
#     if rule["rule_label"] == "benign":
#         return {
#             "schema_version": 1,
#             "query_id": entry.get("query_id"),
#             "timestamp": entry.get("timestamp"),
#             "query_text": query,
#             "final_label": "benign",
#             "confidence": rule["rule_confidence"],
#             "reason": "rule_based_benign_no_llm_needed",
#             "should_escalate": False,
#             "rule_label": rule["rule_label"],
#             "rule_confidence": rule["rule_confidence"],
#             "llm_used": False,
#             "signals": rule["signals"],
#             "rule_reasons": rule["rule_reasons"],
#         }

#     pipeline = RAGPipeline(cfg)
#     pipeline.load_index()
#     retrieved = pipeline.retrieve(query, k=safe_int(entry.get("top_k"), 5))

#     prompt = build_gatekeeper_prompt(entry, retrieved, rule)

#     # Use the model configured in cfg.llm.model_name.
#     llm = LLMGenerator(
#         model_name=cfg.llm.model_name,
#         max_new_tokens=96,
#         temperature=0.0,
#     )
#     raw = llm.generate(prompt)
#     parsed = extract_json_object(raw)

#     # Strong cases stay probing even if the LLM tries to soften them.
#     strong_case = bool(marker_hits > 0 and canary_count >= 2 and top1_is_canary)

#     if parsed is None:
#         final_label = rule["rule_label"]
#         final_confidence = rule["rule_confidence"]
#         reason = "llm_parse_failed_fallback_to_rule_based"
#         should_escalate = True
#     else:
#         final_label = str(parsed.get("final_label", rule["rule_label"])).strip().lower()
#         if final_label not in {"benign", "probing"}:
#             final_label = rule["rule_label"]

#         try:
#             final_confidence = float(parsed.get("confidence", rule["rule_confidence"]))
#         except (TypeError, ValueError):
#             final_confidence = rule["rule_confidence"]

#         reason = str(parsed.get("reason", "")).strip() or "llm_decision"
#         should_escalate = bool(parsed.get("should_escalate", False))

#         # False-positive reduction: allow educational mentions to be downgraded,
#         # but do not downgrade strong canary hits.
#         if strong_case:
#             final_label = "probing"
#             final_confidence = max(final_confidence, rule["rule_confidence"])
#             should_escalate = True
#             reason = reason or "strong_canary_retrieval"

#         elif marker_hits > 0 and looks_educational(query) and canary_count <= 1:
#             final_label = "benign"
#             should_escalate = False

#     return {
#         "schema_version": 1,
#         "query_id": entry.get("query_id"),
#         "timestamp": entry.get("timestamp"),
#         "query_text": query,
#         "final_label": final_label,
#         "confidence": round(max(0.0, min(1.0, final_confidence)), 3),
#         "reason": reason,
#         "should_escalate": should_escalate,
#         "rule_label": rule["rule_label"],
#         "rule_confidence": rule["rule_confidence"],
#         "llm_used": True,
#         "llm_raw": raw[:1000],
#         "signals": rule["signals"],
#         "rule_reasons": rule["rule_reasons"],
#     }


# def main() -> None:
#     parser = argparse.ArgumentParser(description="End-to-end detection pipeline for CanaryRAG.")
#     parser.add_argument("--log-path", type=str, default="artifacts/retrieval_logs.jsonl")
#     parser.add_argument("--query-id", type=str, default=None)
#     parser.add_argument("--latest", action="store_true")
#     parser.add_argument("--output-path", type=str, default="artifacts/final_detections.jsonl")
#     args = parser.parse_args()

#     cfg = RAGConfig()
#     entries = load_jsonl(Path(args.log_path))
#     entries = select_entries(entries, args.query_id, args.latest)

#     if not entries:
#         print("No log entries found.")
#         return

#     results = [run_gatekeeper(entry, cfg) for entry in entries]

#     for result in results:
#         print(json.dumps(result, indent=2, ensure_ascii=False))

#     out_path = Path(args.output_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_path.open("a", encoding="utf-8") as f:
#         for result in results:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")


# if __name__ == "__main__":
#     main()
