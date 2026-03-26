from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorStoreConfig:
    index_path: Path = Path("artifacts/faiss_index.bin")
    metadata_path: Path = Path("artifacts/metadata.jsonl")
    dim: int | None = None  # Will be inferred from embeddings if None
    n_list: int = 100


@dataclass
class LLMConfig:
    # A slightly larger causal LM than distilgpt2 to reduce repetition.
    # You can change this to any HF causal text model string.
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 256
    temperature: float = 0.7


@dataclass
class DataConfig:
    corpus_dir: Path = Path("data/corpus")
    canary_dir: Path = Path("data/canaries")
    combined_dir: Path = Path("data/combined")
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class CanaryConfig:
    keyword_token: str = "ALPHA-CANARY-KEYWORD-123"
    semantic_token: str = "OMEGA-CANARY-SEMANTIC-987"


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    canary: CanaryConfig = field(default_factory=CanaryConfig)
