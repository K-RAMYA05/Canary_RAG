from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# macOS commonly loads multiple OpenMP runtimes when FAISS and PyTorch-based
# dependencies coexist in one process. Allowing duplicate initialization avoids
# the hard abort during local CLI usage on this prototype.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:
    faiss = None


@dataclass
class VectorMetadata:
    doc_id: str
    chunk_id: int
    is_canary: bool
    canary_type: str | None
    text: str


class FAISSVectorStore:
    """
    Vector store with a real FAISS backend when available.

    If `faiss-cpu` is not installed, the store falls back to the existing
    NumPy cosine-similarity implementation so the rest of the project keeps
    working without native dependencies.
    """

    def __init__(
        self,
        dim: int,
        n_list: int,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        self.dim = dim
        self.n_list = n_list
        self.index_path = index_path
        self.metadata_path = metadata_path

        self._backend = "faiss" if faiss is not None else "numpy"
        self._embeddings: np.ndarray | None = None
        self._metadata: list[VectorMetadata] = []
        self._index = None

    @property
    def backend_name(self) -> str:
        return self._backend

    def add(self, embeddings: np.ndarray, metadatas: Sequence[VectorMetadata]) -> None:
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Number of embeddings and metadata entries must match.")

        normalized = self._normalize(embeddings.astype("float32"))
        self._metadata = list(metadatas)

        if self._backend == "faiss":
            self._embeddings = normalized
            self._index = self._build_faiss_index(normalized)
        else:
            self._embeddings = normalized
            self._index = None

    def search(self, query_embeddings: np.ndarray, k: int = 5):
        query = self._normalize(query_embeddings.astype("float32"))
        if self._backend == "faiss":
            if self._index is None or self._index.ntotal == 0:
                raise RuntimeError("Index is empty. Build it before querying.")
            return self._search_faiss(query, k)
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            raise RuntimeError("Index is empty. Build it before querying.")
        return self._search_numpy(query, k)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self._embeddings is None:
            raise RuntimeError("Cannot save an uninitialized index.")

        if self._backend == "faiss":
            if self._index is None:
                raise RuntimeError("FAISS index is not initialized.")
            faiss.write_index(self._index, str(self.index_path))
        else:
            np.save(self.index_path, self._embeddings)

        with self.metadata_path.open("w", encoding="utf-8") as handle:
            for meta in self._metadata:
                handle.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")

    def load(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata file not found.")

        self._metadata = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                self._metadata.append(VectorMetadata(**raw))

        if self._backend == "faiss":
            if not self.index_path.exists():
                raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            self._embeddings = None
            return

        alt = Path(str(self.index_path) + ".npy")
        if alt.exists():
            path = alt
        elif self.index_path.exists() and self.index_path.suffix == ".npy":
            path = self.index_path
        else:
            raise FileNotFoundError(
                "NumPy index file not found. Rebuild the index with the current backend or install faiss-cpu."
            )

        self._embeddings = np.load(path)
        self._index = None

    def _build_faiss_index(self, embeddings: np.ndarray):
        assert faiss is not None
        num_vectors = embeddings.shape[0]

        if num_vectors < max(self.n_list, 40):
            index = faiss.IndexFlatIP(self.dim)
            index.add(embeddings)
            return index

        quantizer = faiss.IndexFlatIP(self.dim)
        nlist = min(self.n_list, num_vectors)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(max(1, nlist // 4), nlist)
        return index

    def _search_faiss(self, query_embeddings: np.ndarray, k: int):
        assert self._index is not None
        k = min(k, len(self._metadata))
        scores, indices = self._index.search(query_embeddings, k)

        results = []
        for row_scores, row_indices in zip(scores, indices):
            row_results = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                row_results.append(
                    {
                        "score": float(score),
                        "metadata": self._metadata[int(idx)],
                    }
                )
            results.append(row_results)
        return results

    def _search_numpy(self, query_embeddings: np.ndarray, k: int):
        assert self._embeddings is not None
        sims = query_embeddings @ self._embeddings.T
        k = min(k, sims.shape[1])
        indices = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row_order = np.argsort(-sims[np.arange(sims.shape[0])[:, None], indices], axis=1)
        indices = indices[np.arange(indices.shape[0])[:, None], row_order]

        results = []
        for i, row in enumerate(indices):
            row_results = []
            for idx in row:
                if idx < 0 or idx >= len(self._metadata):
                    continue
                row_results.append(
                    {
                        "score": float(sims[i, int(idx)]),
                        "metadata": self._metadata[int(idx)],
                    }
                )
            results.append(row_results)
        return results

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
