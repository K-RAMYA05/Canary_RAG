from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.pipeline import RAGPipeline
from canaryrag.logging import build_log_entry, append_jsonl


def load_query_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query dataset not found: {path}")

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


def summarize_retrievals(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """
    Lightweight summary metrics over retrieval logs for quick inspection.
    """
    entries = list(entries)
    total = len(entries)
    if total == 0:
        return {
            "total_queries": 0,
            "overall_canary_hit_rate": 0.0,
            "benign_canary_hit_rate": None,
            "probing_canary_hit_rate": None,
        }

    canary_hits = 0
    top1_canary_hits = 0

    per_label: dict[str, dict[str, int]] = {}
    per_group: dict[str, dict[str, int]] = {}

    for e in entries:
        has_canary = int(e.get("canary_count", 0)) > 0
        top1_is_canary = bool(e.get("top1_is_canary"))
        if has_canary:
            canary_hits += 1
        if top1_is_canary:
            top1_canary_hits += 1

        label = str(e.get("label", "")).lower() or "unlabeled"
        group = str(e.get("group", "")).lower() or "ungrouped"

        for bucket, key in ((per_label, label), (per_group, group)):
            stats = bucket.setdefault(key, {"total": 0, "canary_hits": 0, "top1_canary_hits": 0})
            stats["total"] += 1
            if has_canary:
                stats["canary_hits"] += 1
            if top1_is_canary:
                stats["top1_canary_hits"] += 1

    def _rates(stats: dict[str, int]) -> dict[str, float]:
        total_local = stats.get("total", 0)
        ch = stats.get("canary_hits", 0)
        top1 = stats.get("top1_canary_hits", 0)
        return {
            "total": total_local,
            "canary_hit_rate": ch / total_local if total_local else 0.0,
            "top1_canary_rate": top1 / total_local if total_local else 0.0,
        }

    overall_rate = canary_hits / total
    overall_top1_rate = top1_canary_hits / total

    by_label_rates = {k: _rates(v) for k, v in per_label.items()}
    by_group_rates = {k: _rates(v) for k, v in per_group.items()}

    return {
        "total_queries": total,
        "overall": {
            "canary_hit_rate": overall_rate,
            "top1_canary_rate": overall_top1_rate,
        },
        "by_label": by_label_rates,
        "by_group": by_group_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of queries through CanaryRAG and log retrievals.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(ROOT / "data" / "queries.jsonl"),
        help="Path to labeled query dataset (JSONL).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(ROOT / "artifacts" / "retrieval_logs.jsonl"),
        help="Path to append retrieval logs (JSONL).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved chunks per query.",
    )
    parser.add_argument(
        "--overwrite-log",
        action="store_true",
        help="If set, truncate the log file before running.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    log_path = Path(args.log_path)

    queries = load_query_dataset(dataset_path)

    if args.overwrite_log and log_path.exists():
        log_path.unlink()

    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()

    canary_tokens = [cfg.canary.keyword_token, cfg.canary.semantic_token]

    enriched_entries: list[dict[str, Any]] = []

    for row in queries:
        text = str(row.get("query_text", "")).strip()
        if not text:
            continue
        label = row.get("label")
        group = row.get("group")
        user_id = row.get("user_id")
        session_id = row.get("session_id")

        retrieved = pipeline.retrieve(text, k=args.k)
        log_entry = build_log_entry(
            query_text=text,
            retrieved=retrieved,
            top_k=args.k,
            canary_tokens=canary_tokens,
        )

        # Attach ground-truth metadata for downstream evaluation and attribution.
        if label is not None:
            log_entry["label"] = label
        if group is not None:
            log_entry["group"] = group
        if user_id is not None:
            log_entry["user_id"] = user_id
        if session_id is not None:
            log_entry["session_id"] = session_id

        append_jsonl(log_path, log_entry)
        enriched_entries.append(log_entry)

    summary = summarize_retrievals(enriched_entries)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
