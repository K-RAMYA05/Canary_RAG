from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.canary import CanaryDesign, write_canary_corpus
from canaryrag.config import RAGConfig


def main() -> None:
    cfg = RAGConfig()
    design = CanaryDesign(
        keyword_token=cfg.canary.keyword_token,
        semantic_token=cfg.canary.semantic_token,
    )
    output_dir = cfg.data.canary_dir
    write_canary_corpus(output_dir, design)
    print(f"Wrote canary corpus to {output_dir}")


if __name__ == "__main__":
    main()
