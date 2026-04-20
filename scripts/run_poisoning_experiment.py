from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.logging import append_jsonl, build_log_entry
from canaryrag.pipeline import RAGPipeline
from scripts.run_experiments import load_query_dataset, summarize_retrievals
from scripts.simulate_poisoning import copy_corpus, materialize_poisoned_docs


def build_config(corpus_dir: Path, index_path: Path, metadata_path: Path) -> RAGConfig:
    cfg = RAGConfig()
    cfg.data.corpus_dir = corpus_dir
    cfg.vector_store.index_path = index_path
    cfg.vector_store.metadata_path = metadata_path
    return cfg


def build_index(cfg: RAGConfig) -> None:
    pipeline = RAGPipeline(cfg)
    pipeline.build_index(include_canaries=True)


def enrich_with_poison_signals(entry: dict[str, Any]) -> dict[str, Any]:
    retrieved = entry.get("retrieved", [])
    poisoned_items = [item for item in retrieved if str(item.get("doc_id", "")).startswith("poisoned_")]
    poisoned_count = len(poisoned_items)
    top1_is_poisoned = bool(retrieved and str(retrieved[0].get("doc_id", "")).startswith("poisoned_"))
    best_poisoned_rank = poisoned_items[0]["rank"] if poisoned_items else None
    score_gap_top1_vs_top_poisoned = None
    if poisoned_items and retrieved:
        score_gap_top1_vs_top_poisoned = retrieved[0]["score"] - poisoned_items[0]["score"]

    entry["poisoned_count"] = poisoned_count
    entry["top1_is_poisoned"] = top1_is_poisoned
    entry["best_poisoned_rank"] = best_poisoned_rank
    entry["score_gap_top1_vs_top_poisoned"] = score_gap_top1_vs_top_poisoned
    return entry


def run_dataset(
    cfg: RAGConfig,
    dataset_path: Path,
    log_path: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    if log_path.exists():
        log_path.unlink()

    queries = load_query_dataset(dataset_path)
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()
    canary_tokens = [cfg.canary.keyword_token, cfg.canary.semantic_token]

    rows: list[dict[str, Any]] = []

    for row in queries:
        text = str(row.get("query_text", "")).strip()
        if not text:
            continue

        retrieved = pipeline.retrieve(text, k=top_k)
        entry = build_log_entry(
            query_text=text,
            retrieved=retrieved,
            top_k=top_k,
            canary_tokens=canary_tokens,
        )
        enrich_with_poison_signals(entry)

        for key in ("label", "group", "user_id", "session_id"):
            if row.get(key) is not None:
                entry[key] = row[key]

        append_jsonl(log_path, entry)
        rows.append(entry)

    return rows


def summarize_poisoning(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    poisoned_hits = sum(1 for e in entries if int(e.get("poisoned_count", 0)) > 0)
    top1_poisoned = sum(1 for e in entries if bool(e.get("top1_is_poisoned")))

    by_label: dict[str, dict[str, int]] = {}
    by_group: dict[str, dict[str, int]] = {}

    for e in entries:
        label = str(e.get("label", "")).lower() or "unlabeled"
        group = str(e.get("group", "")).lower() or "ungrouped"

        for bucket, key in ((by_label, label), (by_group, group)):
            stats = bucket.setdefault(key, {"total": 0, "poisoned_hits": 0, "top1_poisoned": 0})
            stats["total"] += 1
            if int(e.get("poisoned_count", 0)) > 0:
                stats["poisoned_hits"] += 1
            if bool(e.get("top1_is_poisoned")):
                stats["top1_poisoned"] += 1

    def rates(stats: dict[str, int]) -> dict[str, float]:
        total_local = stats["total"]
        return {
            "total": total_local,
            "poisoned_hit_rate": stats["poisoned_hits"] / total_local if total_local else 0.0,
            "top1_poisoned_rate": stats["top1_poisoned"] / total_local if total_local else 0.0,
        }

    return {
        "total_queries": total,
        "overall": {
            "poisoned_hit_rate": poisoned_hits / total if total else 0.0,
            "top1_poisoned_rate": top1_poisoned / total if total else 0.0,
        },
        "by_label": {k: rates(v) for k, v in by_label.items()},
        "by_group": {k: rates(v) for k, v in by_group.items()},
    }


def compare_runs(clean_entries: list[dict[str, Any]], poisoned_entries: list[dict[str, Any]]) -> dict[str, Any]:
    clean_by_text = {str(e["query_text"]): e for e in clean_entries}
    poisoned_by_text = {str(e["query_text"]): e for e in poisoned_entries}

    comparisons = []
    top1_changed = 0
    canary_increase = 0
    poisoned_top1_gain = 0

    for query_text, poisoned_entry in poisoned_by_text.items():
        clean_entry = clean_by_text.get(query_text)
        if clean_entry is None:
            continue

        clean_top1 = clean_entry["retrieved"][0]["doc_id"] if clean_entry.get("retrieved") else None
        poisoned_top1 = poisoned_entry["retrieved"][0]["doc_id"] if poisoned_entry.get("retrieved") else None
        top1_changed_flag = clean_top1 != poisoned_top1
        if top1_changed_flag:
            top1_changed += 1

        clean_canary = int(clean_entry.get("canary_count", 0))
        poisoned_canary = int(poisoned_entry.get("canary_count", 0))
        if poisoned_canary > clean_canary:
            canary_increase += 1

        if bool(poisoned_entry.get("top1_is_poisoned")) and not bool(clean_entry.get("top1_is_poisoned")):
            poisoned_top1_gain += 1

        comparisons.append(
            {
                "query_text": query_text,
                "label": poisoned_entry.get("label"),
                "group": poisoned_entry.get("group"),
                "clean_top1": clean_top1,
                "poisoned_top1": poisoned_top1,
                "top1_changed": top1_changed_flag,
                "clean_canary_count": clean_canary,
                "poisoned_canary_count": poisoned_canary,
                "poisoned_count_in_poisoned_run": poisoned_entry.get("poisoned_count", 0),
                "top1_is_poisoned": poisoned_entry.get("top1_is_poisoned", False),
            }
        )

    total = len(comparisons)
    return {
        "total_compared_queries": total,
        "top1_changed_rate": top1_changed / total if total else 0.0,
        "queries_with_higher_canary_count_after_poisoning": canary_increase,
        "queries_with_poisoned_doc_at_top1_only_after_poisoning": poisoned_top1_gain,
        "sample_comparisons": comparisons[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean and poisoned indexes and compare retrieval behavior.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(ROOT / "data" / "queries.jsonl"),
        help="Path to labeled query dataset.",
    )
    parser.add_argument(
        "--clean-corpus-dir",
        type=str,
        default=str(ROOT / "data" / "corpus"),
        help="Path to the clean corpus directory.",
    )
    parser.add_argument(
        "--poisoned-corpus-dir",
        type=str,
        default=str(ROOT / "data" / "poisoned_corpus"),
        help="Path to the poisoned corpus directory.",
    )
    parser.add_argument(
        "--docs-per-type",
        type=int,
        default=2,
        help="Number of keyword-poison and semantic-poison docs to synthesize per document type.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k retrieval size.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    clean_corpus_dir = Path(args.clean_corpus_dir)
    poisoned_corpus_dir = Path(args.poisoned_corpus_dir)

    cfg = RAGConfig()
    copy_corpus(clean_corpus_dir, poisoned_corpus_dir)
    materialize_poisoned_docs(
        output_dir=poisoned_corpus_dir,
        keyword_token=cfg.canary.keyword_token,
        semantic_token=cfg.canary.semantic_token,
        docs_per_type=args.docs_per_type,
    )

    clean_cfg = build_config(
        corpus_dir=clean_corpus_dir,
        index_path=ROOT / "artifacts" / "poisoning_clean_index.bin",
        metadata_path=ROOT / "artifacts" / "poisoning_clean_metadata.jsonl",
    )
    poisoned_cfg = build_config(
        corpus_dir=poisoned_corpus_dir,
        index_path=ROOT / "artifacts" / "poisoning_index.bin",
        metadata_path=ROOT / "artifacts" / "poisoning_metadata.jsonl",
    )

    build_index(clean_cfg)
    build_index(poisoned_cfg)

    clean_log = ROOT / "artifacts" / "poisoning_clean_retrieval_logs.jsonl"
    poisoned_log = ROOT / "artifacts" / "poisoning_retrieval_logs.jsonl"

    clean_entries = run_dataset(clean_cfg, dataset_path, clean_log, args.k)
    poisoned_entries = run_dataset(poisoned_cfg, dataset_path, poisoned_log, args.k)

    summary = {
        "clean_retrieval": summarize_retrievals(clean_entries),
        "poisoned_retrieval": summarize_retrievals(poisoned_entries),
        "poisoning_specific": summarize_poisoning(poisoned_entries),
        "comparison": compare_runs(clean_entries, poisoned_entries),
        "poisoned_corpus_dir": str(poisoned_corpus_dir),
        "clean_log_path": str(clean_log),
        "poisoned_log_path": str(poisoned_log),
    }

    summary_path = ROOT / "artifacts" / "poisoning_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
