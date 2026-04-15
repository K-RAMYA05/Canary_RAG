from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import RAGConfig
from .embeddings import EmbeddingModel
from .vectorstore import FAISSVectorStore, VectorMetadata


def read_text_files(root: Path) -> list[tuple[str, str]]:
    """
    Recursively read all .txt files under a directory.
    Returns list of (doc_id, text).
    """
    docs: list[tuple[str, str]] = []
    if not root.exists():
        return docs
    for path in root.rglob("*.txt"):
        doc_id = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append((doc_id, text))
    return docs


def simple_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Overlapping character-based chunking.
    """
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


@dataclass
class RAGPipeline:
    config: RAGConfig

    def __post_init__(self) -> None:
        self._embedder = EmbeddingModel(
            self.config.embedding.model_name,
            keyword_token=self.config.canary.keyword_token,
            semantic_token=self.config.canary.semantic_token,
        )
        dim = self.config.vector_store.dim or self._embedder.dim
        self._store = FAISSVectorStore(
            dim=dim,
            n_list=self.config.vector_store.n_list,
            index_path=self.config.vector_store.index_path,
            metadata_path=self.config.vector_store.metadata_path,
        )
        # Lazily initialize the LLM only when text generation is needed.
        # This prevents loading large models (and potential segfaults) during
        # index building or retrieval-only tasks such as sanity checks.
        self._llm = None  # type: ignore[assignment]

    def _ensure_llm(self) -> None:
        """
        Import and initialize the LLM generator on first use.

        Kept separate so that building the index and running retrieval do not
        require the transformers stack to be installed.
        """
        if self._llm is None:
            from .llm import LLMGenerator  # local import to avoid hard dependency

            self._llm = LLMGenerator(
                model_name=self.config.llm.model_name,
                max_new_tokens=self.config.llm.max_new_tokens,
                temperature=self.config.llm.temperature,
            )

    def build_index(self, include_canaries: bool = True) -> None:
        """
        Ingest corpus + optional canaries, build FAISS index, and persist to disk.
        """
        texts: list[str] = []
        metadatas: list[VectorMetadata] = []

        # Regular corpus.
        corpus_docs = read_text_files(self.config.data.corpus_dir)
        for doc_id, text in corpus_docs:
            chunks = simple_chunk(
                text,
                chunk_size=self.config.data.chunk_size,
                chunk_overlap=self.config.data.chunk_overlap,
            )
            for chunk_id, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append(
                    VectorMetadata(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        is_canary=False,
                        canary_type=None,
                        text=chunk,
                    )
                )

        # Canary corpus (if enabled).
        if include_canaries:
            canary_docs = read_text_files(self.config.data.canary_dir)
            for doc_id, text in canary_docs:
                chunks = simple_chunk(
                    text,
                    chunk_size=self.config.data.chunk_size,
                    chunk_overlap=self.config.data.chunk_overlap,
                )
                for chunk_id, chunk in enumerate(chunks):
                    is_keyword = "keyword_canary" in doc_id
                    is_semantic = "semantic_canary" in doc_id
                    canary_type: str | None
                    if is_keyword:
                        canary_type = "keyword"
                    elif is_semantic:
                        canary_type = "semantic"
                    else:
                        canary_type = "unknown"
                    texts.append(chunk)
                    metadatas.append(
                        VectorMetadata(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            is_canary=True,
                            canary_type=canary_type,
                            text=chunk,
                        )
                    )

        if not texts:
            raise RuntimeError("No documents found to build the index.")

        embeddings = self._embedder.embed(texts)
        self._store.add(embeddings, metadatas)
        self._store.save()

    def load_index(self) -> None:
        self._store.load()

    def retrieve(self, query: str, k: int = 5):
        query_emb = self._embedder.embed([query])
        results = self._store.search(query_emb, k=k)[0]
        return results

    def generate_answer(self, query: str, k: int = 5) -> dict:
        """
        Retrieve top-k documents and generate an answer using the LLM.
        Returns both the answer and the retrieval context.
        """
        retrieved = self.retrieve(query, k=k)
        context_blocks = []
        for item in retrieved:
            meta = item["metadata"]
            prefix = ""
            if meta.is_canary:
                prefix = "[CANARY] "
            context_blocks.append(f"{prefix}{meta.text}")
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a helpful assistant answering questions based only on the provided context.\n"
            "If the answer is not present in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        self._ensure_llm()
        assert self._llm is not None
        answer = self._llm.generate(prompt)
        return {
            "answer": answer,
            "retrieved": retrieved,
        }
