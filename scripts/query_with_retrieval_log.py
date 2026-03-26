from __future__ import annotations

import argparse
import json
import re
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.pipeline import RAGPipeline


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+", text.lower())


def _count_repetition(text: str) -> int:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    return sum(count - 1 for count in counts.values() if count > 1)


def _count_marker_hits(text: str, markers: list[str]) -> int:
    q = text.lower()
    return sum(q.count(marker.lower()) for marker in markers)


def build_log_entry(
    query_text: str,
    retrieved: list[dict],
    top_k: int,
    canary_tokens: list[str],
    query_id: str | None = None,
) -> dict:
    qid = query_id or uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()

    retrieved_items: list[dict] = []
    for rank, item in enumerate(retrieved, start=1):
        meta = item["metadata"]
        retrieved_items.append(
            {
                "rank": rank,
                "score": float(item["score"]),
                "doc_id": meta.doc_id,
                "chunk_id": meta.chunk_id,
                "is_canary": bool(meta.is_canary),
                "canary_type": meta.canary_type,
            }
        )

    canary_items = [x for x in retrieved_items if x["is_canary"]]
    canary_count = len(canary_items)
    top1_is_canary = bool(retrieved_items and retrieved_items[0]["is_canary"])

    best_canary_rank = canary_items[0]["rank"] if canary_items else None
    score_gap_top1_vs_top_canary = None
    if canary_items and retrieved_items:
        score_gap_top1_vs_top_canary = retrieved_items[0]["score"] - canary_items[0]["score"]

    entry = {
        "schema_version": 1,
        "query_id": qid,
        "timestamp": timestamp,
        "query_text": query_text,
        "top_k": top_k,
        "retrieved": retrieved_items,
        "canary_count": canary_count,
        "top1_is_canary": top1_is_canary,
        "best_canary_rank": best_canary_rank,
        "score_gap_top1_vs_top_canary": score_gap_top1_vs_top_canary,
        "repetition_count": _count_repetition(query_text),
        "marker_hits": _count_marker_hits(query_text, canary_tokens),
    }
    return entry


def append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query CanaryRAG and log retrievals.")
    parser.add_argument("question", type=str, help="User question to ask the RAG system.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks.")
    parser.add_argument(
        "--log-path",
        type=str,
        default="artifacts/retrieval_logs.jsonl",
        help="Path to JSONL retrieval log file.",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Only log retrieval results; do not generate an answer.",
    )
    args = parser.parse_args()

    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()

    # Retrieve once for logging and inspection.
    retrieved = pipeline.retrieve(args.question, k=args.k)

    log_entry = build_log_entry(
        query_text=args.question,
        retrieved=retrieved,
        top_k=args.k,
        canary_tokens=[cfg.canary.keyword_token, cfg.canary.semantic_token],
    )
    append_jsonl(Path(args.log_path), log_entry)

    print("\n=== Retrieval Log Entry ===\n")
    print(json.dumps(log_entry, indent=2, ensure_ascii=False))

    print("\n=== Retrieved Chunks ===\n")
    for item in log_entry["retrieved"]:
        print(
            f"[{item['rank']}] score={item['score']:.3f} "
            f"doc_id={item['doc_id']} chunk_id={item['chunk_id']} "
            f"canary={item['is_canary']} type={item['canary_type']}"
        )

    # if args.no_answer:
    #     return

    # # This uses the existing pipeline as-is.
    # answer_out = pipeline.generate_answer(args.question, k=args.k)
    # print("\n=== Answer ===\n")
    # print(answer_out["answer"])


if __name__ == "__main__":
    main()