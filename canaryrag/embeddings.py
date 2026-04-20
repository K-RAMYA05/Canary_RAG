from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np


class EmbeddingModel:
    """
    Simple hashing-based embedding model with optional special dimensions
    for canary marker tokens.

    This avoids heavy model downloads and potential segfaults from GPU/BLAS
    stacks, while still providing deterministic float32 vectors suitable for
    vector search.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dim: int = 384,
        keyword_token: str | None = None,
        semantic_token: str | None = None,
        semantic_probe_terms: tuple[str, ...] | None = None,
    ) -> None:
        # model_name is accepted for API compatibility but ignored.
        self._base_dim = dim

        # Reserve dedicated dimensions for canary markers so that
        # queries mentioning them strongly align with canary documents
        # but have no effect on benign queries.
        specials: dict[str, int] = {}
        next_idx = self._base_dim

        if keyword_token:
            specials[keyword_token.lower()] = next_idx
            next_idx += 1
        if semantic_token and semantic_token.lower() not in specials:
            specials[semantic_token.lower()] = next_idx
            next_idx += 1

        probe_terms = semantic_probe_terms or (
            "synthetic",
            "protocol",
            "identifier",
            "identifiers",
            "registry",
            "hidden",
            "least",
            "accessed",
            "artifact",
            "artifacts",
            "dormant",
        )
        probe_indices: dict[str, int] = {}
        for term in probe_terms:
            normalized = term.lower()
            if normalized in specials or normalized in probe_indices:
                continue
            probe_indices[normalized] = next_idx
            next_idx += 1

        self._special_indices = specials
        self._semantic_probe_indices = probe_indices
        self._dim = next_idx

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """
        Very simple bag-of-words hashing embedding with extra dimensions
        for specific canary marker tokens:

        - Lowercase, whitespace tokenization.
        - Each token is mapped to a "base" index via Python's hash.
        - If the token matches a special marker, it also updates a dedicated
          high-weight dimension.
        - Counts per dimension, then L2-normalized.
        """
        texts = list(texts)
        n = len(texts)
        dim = self._dim
        base_dim = self._base_dim
        mat = np.zeros((n, dim), dtype="float32")

        for i, text in enumerate(texts):
            for token in _tokenize(text):
                # Base hashed dimension.
                idx = _stable_hash_index(token, base_dim)
                mat[i, idx] += 1.0

                # Optional special canary dimension.
                special_idx = self._special_indices.get(token)
                if special_idx is not None:
                    # Use a higher weight so the marker dominates similarity
                    # when present, without affecting benign queries.
                    mat[i, special_idx] += 5.0

                probe_idx = self._semantic_probe_indices.get(token)
                if probe_idx is not None:
                    # Give semantically suspicious probe concepts a smaller,
                    # controlled boost so semantic canaries can align with
                    # queries about hidden test artifacts and marker registries
                    # without broadly changing benign retrieval behavior.
                    mat[i, probe_idx] += 2.5

        # L2 normalize rows where possible.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        mat[nonzero] /= norms[nonzero]
        return mat


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+", text.lower())


def _stable_hash_index(token: str, base_dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % base_dim
