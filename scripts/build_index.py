from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.pipeline import RAGPipeline


def main() -> None:
    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.build_index(include_canaries=True)
    print("Index built and saved.")


if __name__ == "__main__":
    main()
