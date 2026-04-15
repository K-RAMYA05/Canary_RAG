from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.pipeline import RAGPipeline
from canaryrag.logging import build_log_entry, append_jsonl


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
