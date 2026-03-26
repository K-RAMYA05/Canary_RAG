from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOP_CANARY_RANK_THRESHOLD = 3


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


def classify_entry(entry: dict[str, Any]) -> dict[str, Any]:
    query_text = str(entry.get("query_text", ""))
    marker_hits = safe_int(entry.get("marker_hits"), 0)
    canary_count = safe_int(entry.get("canary_count"), 0)
    top1_is_canary = safe_bool(entry.get("top1_is_canary"))
    best_canary_rank_raw = entry.get("best_canary_rank")
    best_canary_rank = None if best_canary_rank_raw is None else safe_int(best_canary_rank_raw, 0)

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

        confidence = 0.70
        confidence += min(marker_hits, 3) * 0.08
        confidence += min(canary_count, 3) * 0.05
        if top1_is_canary:
            confidence += 0.10
        confidence = min(confidence, 0.99)

    elif canary_count > 0:
        label = "poisoning"
        reasons.append("canary_retrieved_without_query_marker")

        if top1_is_canary:
            reasons.append("canary_retrieved_at_rank_1")
        if best_canary_rank is not None and best_canary_rank <= TOP_CANARY_RANK_THRESHOLD:
            reasons.append(f"canary_in_top_{TOP_CANARY_RANK_THRESHOLD}")

        confidence = 0.75
        confidence += min(canary_count, 5) * 0.05
        if top1_is_canary:
            confidence += 0.10
        confidence = min(confidence, 0.99)

    else:
        label = "benign"
        reasons.append("no_canary_marker_in_query")
        reasons.append("no_canary_in_retrieval")
        confidence = 0.97

    return {
        "schema_version": 1,
        "query_id": entry.get("query_id"),
        "timestamp": entry.get("timestamp"),
        "query_text": query_text,
        "label": label,
        "confidence": round(confidence, 3),
        "signals": {
            "marker_hits": marker_hits,
            "canary_count": canary_count,
            "top1_is_canary": top1_is_canary,
            "best_canary_rank": best_canary_rank,
        },
        "reasons": reasons,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based detector for CanaryRAG retrieval logs.")
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
        help="Detect only one log entry by query_id.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Detect only the most recent log entry.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write detection results as JSONL.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    entries = load_jsonl(log_path)
    entries = select_entries(entries, args.query_id, args.latest)

    results = [classify_entry(entry) for entry in entries]

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