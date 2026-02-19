from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class VectorMetadata:
    doc_id: str
    chunk_id: int
    is_canary: bool
    canary_type: str | None  # "keyword", "semantic", or None
    text: str


class FAISSVectorStore:
    """
    Lightweight in-memory vector store with JSONL metadata.

    The class name is kept for compatibility, but implementation uses pure
    NumPy instead of FAISS to avoid native-library issues.
    """

    def __init__(
        self,
        dim: int,
        n_list: int,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        self.dim = dim
        self.n_list = n_list  # kept for API compatibility, unused
        self.index_path = index_path
        self.metadata_path = metadata_path

        self._embeddings: np.ndarray | None = None
        self._metadata: list[VectorMetadata] = []

    def add(self, embeddings: np.ndarray, metadatas: Sequence[VectorMetadata]) -> None:
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Number of embeddings and metadata entries must match.")
        if self._embeddings is None:
            self._embeddings = embeddings.astype("float32")
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings.astype("float32")])
        self._metadata.extend(metadatas)

    def search(self, query_embeddings: np.ndarray, k: int = 5):
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            raise RuntimeError("Index is empty. Build it before querying.")
        # Compute cosine similarity between query and stored embeddings.
        query = query_embeddings.astype("float32")
        # Normalize
        def _l2norm(x: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return x / n

        q_norm = _l2norm(query)
        e_norm = _l2norm(self._embeddings)
        sims = q_norm @ e_norm.T  # (num_queries, num_docs)

        # For each query, get top-k indices.
        k = min(k, sims.shape[1])
        indices = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        # Sort each row by similarity descending.
        row_order = np.argsort(-sims[np.arange(sims.shape[0])[:, None], indices], axis=1)
        indices = indices[np.arange(indices.shape[0])[:, None], row_order]

        results = []
        for i, row in enumerate(indices):
            row_results = []
            for j, idx in enumerate(row):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                meta = self._metadata[int(idx)]
                row_results.append(
                    {
                        "score": float(sims[i, int(idx)]),
                        "metadata": meta,
                    }
                )
            results.append(row_results)
        return results

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self._embeddings is None:
            raise RuntimeError("Cannot save an uninitialized index.")

        # Save embeddings as a simple .npy file
        np.save(self.index_path, self._embeddings)
        with self.metadata_path.open("w", encoding="utf-8") as f:
            for meta in self._metadata:
                f.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")

    def load(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata file not found.")

        path = self.index_path
        if not path.exists():
            alt = Path(str(path) + ".npy")
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError("Index or metadata file not found.")

        self._embeddings = np.load(path)
        self._metadata = []
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                self._metadata.append(VectorMetadata(**raw))
