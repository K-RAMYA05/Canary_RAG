from __future__ import annotations

from typing import Iterable

import numpy as np


class EmbeddingModel:
    """
    Simple hashing-based embedding model with no external ML dependencies.

    This avoids heavy model downloads and potential segfaults from GPU/BLAS
    stacks, while still providing deterministic float32 vectors suitable for
    FAISS indexing.
    """

    def __init__(self, model_name: str | None = None, dim: int = 384) -> None:
        # model_name is accepted for API compatibility but ignored.
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        import re
        """
        Very simple bag-of-words hashing embedding:

        - Lowercase, whitespace tokenization.
        - Each token is mapped to an index via Python's hash.
        - Counts per dimension, then L2-normalized.
        """
        texts = list(texts)
        n = len(texts)
        dim = self._dim
        mat = np.zeros((n, dim), dtype="float32")

        for i, text in enumerate(texts):
            for token in text.lower().split():
                idx = hash(token) % dim
                mat[i, idx] += 1.0

        # L2 normalize rows where possible.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        mat[nonzero] /= norms[nonzero]
        return mat
