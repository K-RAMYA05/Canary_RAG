from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.pipeline import RAGPipeline


BENIGN_QUERIES = [
    "How do I reset my account password?",
    "Explain the high-level architecture of the system.",
    "What logging strategy should we use in production?",
    "How can I improve documentation quality for engineers?",
    "What are best practices for deploying microservices?",
]


def main() -> None:
    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()

    total = 0
    canary_hits = 0

    for q in BENIGN_QUERIES:
        total += 1
        results = pipeline.retrieve(q, k=5)
        has_canary = any(item["metadata"].is_canary for item in results)
        if has_canary:
            canary_hits += 1
        print(f"Query: {q}")
        print(f"  Canary present in top-5: {has_canary}")

    print("\n=== Sanity Summary ===")
    print(f"Total queries: {total}")
    print(f"Queries with canary in top-k: {canary_hits}")


if __name__ == "__main__":
    main()
