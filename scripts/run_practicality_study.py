from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.canary import CanaryDesign, write_canary_corpus
from canaryrag.config import RAGConfig
from canaryrag.logging import build_log_entry
from canaryrag.pipeline import RAGPipeline
from scripts.run_experiments import load_query_dataset, summarize_retrievals


def build_config(canary_dir: Path, index_path: Path, metadata_path: Path) -> RAGConfig:
    cfg = RAGConfig()
    cfg.data.canary_dir = canary_dir
    cfg.vector_store.index_path = index_path
    cfg.vector_store.metadata_path = metadata_path
    return cfg


def count_canary_chunks(metadata_path: Path) -> int:
    count = 0
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("is_canary")):
                count += 1
    return count


def run_one_setting(total_canaries: int, dataset_path: Path, top_k: int) -> dict[str, Any]:
    if total_canaries % 2 != 0:
        raise ValueError("Total canary count must be even so keyword and semantic canaries can be balanced.")

    per_type = total_canaries // 2
    canary_dir = ROOT / "artifacts" / "practicality" / f"canaries_{total_canaries}"
    index_base = ROOT / "artifacts" / "practicality" / f"index_{total_canaries}.bin"
    metadata_path = ROOT / "artifacts" / "practicality" / f"metadata_{total_canaries}.jsonl"

    cfg = RAGConfig()
    design = CanaryDesign(
        keyword_token=cfg.canary.keyword_token,
        semantic_token=cfg.canary.semantic_token,
    )
    if canary_dir.exists():
        for p in canary_dir.glob("*.txt"):
            p.unlink()
    write_canary_corpus(canary_dir, design, keyword_count=per_type, semantic_count=per_type)

    local_cfg = build_config(canary_dir=canary_dir, index_path=index_base, metadata_path=metadata_path)
    pipeline = RAGPipeline(local_cfg)

    build_start = time.perf_counter()
    pipeline.build_index(include_canaries=True)
    build_seconds = time.perf_counter() - build_start

    pipeline = RAGPipeline(local_cfg)
    pipeline.load_index()
    queries = load_query_dataset(dataset_path)
    canary_tokens = [local_cfg.canary.keyword_token, local_cfg.canary.semantic_token]

    entries: list[dict[str, Any]] = []
    query_start = time.perf_counter()
    for row in queries:
        query_text = str(row.get("query_text", "")).strip()
        if not query_text:
            continue
        retrieved = pipeline.retrieve(query_text, k=top_k)
        entry = build_log_entry(
            query_text=query_text,
            retrieved=retrieved,
            top_k=top_k,
            canary_tokens=canary_tokens,
        )
        for key in ("label", "group", "user_id", "session_id"):
            if row.get(key) is not None:
                entry[key] = row[key]
        entries.append(entry)
    query_seconds = time.perf_counter() - query_start

    retrieval_summary = summarize_retrievals(entries)
    index_npy = Path(str(index_base) + ".npy")
    arr = np.load(index_npy)

    return {
        "total_canaries": total_canaries,
        "keyword_canaries": per_type,
        "semantic_canaries": per_type,
        "canary_files": len(list(canary_dir.glob("*.txt"))),
        "canary_chunks": count_canary_chunks(metadata_path),
        "embedding_rows": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]),
        "index_bytes": int(index_npy.stat().st_size),
        "metadata_bytes": int(metadata_path.stat().st_size),
        "build_seconds": round(build_seconds, 4),
        "query_batch_seconds": round(query_seconds, 4),
        "avg_query_seconds": round(query_seconds / len(entries), 6) if entries else 0.0,
        "retrieval_summary": retrieval_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a practicality study over different canary counts.")
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[4, 10, 20, 50],
        help="Total canary counts to test. Each count is split evenly between keyword and semantic canaries.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(ROOT / "data" / "queries.jsonl"),
        help="Path to labeled query dataset.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retriever top-k.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(ROOT / "artifacts" / "practicality_study_summary.json"),
        help="Where to write the practicality study summary.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    results = [run_one_setting(count, dataset_path, args.k) for count in args.counts]

    summary = {
        "counts_tested": args.counts,
        "results": results,
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
