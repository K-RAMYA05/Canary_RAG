from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Query CanaryRAG.")
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="User question to ask the RAG system.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks.")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run a few built-in example queries instead of a single question.",
    )
    args = parser.parse_args()

    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()
    if args.example:
        example_questions = [
            "How do I reset my password?",
            "Summarize the latest changes.",
            "What is the incident process?",
            "Tell me about data retention policy.",
        ]
        for q in example_questions:
            out = pipeline.generate_answer(q, k=args.k)
            print(f"\n=== Question ===\n{q}\n")
            print("=== Answer ===\n")
            print(out["answer"])
            print("\n=== Retrieved Chunks (for inspection) ===\n")
            for rank, item in enumerate(out["retrieved"], start=1):
                meta = item["metadata"]
                print(
                    f"[{rank}] score={item['score']:.3f} doc_id={meta.doc_id} "
                    f"canary={meta.is_canary} type={meta.canary_type}"
                )
        return

    if not args.question:
        parser.error("You must provide a question, or use --example.")

    out = pipeline.generate_answer(args.question, k=args.k)

    print("\n=== Answer ===\n")
    print(out["answer"])
    print("\n=== Retrieved Chunks (for inspection) ===\n")
    for rank, item in enumerate(out["retrieved"], start=1):
        meta = item["metadata"]
        print(
            f"[{rank}] score={item['score']:.3f} doc_id={meta.doc_id} "
            f"canary={meta.is_canary} type={meta.canary_type}"
        )


if __name__ == "__main__":
    main()
