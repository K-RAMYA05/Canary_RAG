# CanaryRAG – Baseline RAG + Canary Design

This repository implements **Person 1**'s scope from the project description:

- Baseline RAG pipeline (embeddings + vector DB + open-source LLM)
- Synthetic canary document design and insertion
- Sanity checks for retrieval behavior under benign usage
- Large synthetic benign corpus generation

## Overview

- **Embedding model**: lightweight hashing-based embedder (no external ML dependency)
- **Vector database**: NumPy-based cosine-similarity index with JSONL metadata
- **LLM for generation**: Hugging Face `gpt2` by default (configurable, optional)
- **Canaries**:
  - Keyword-dense canaries for poisoning / boosting detection
  - Semantically general canaries for probing / membership inference
  - Designed to be *irrelevant to benign queries* while remaining
    *structurally indistinguishable* from normal documents.

## Installation

Create a virtual environment and install dependencies:

```bash
cd CanaryRAG
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Note: the first run will download models from Hugging Face
> (`sentence-transformers` and `distilgpt2`), which requires network access.

## Project Layout

- `canaryrag/`
  - `config.py` – central configuration for models, data paths, and canary tokens
  - `embeddings.py` – simple hashing-based embedding model (deterministic float32 vectors)
  - `vectorstore.py` – in-memory cosine-similarity vector store + JSONL metadata
  - `llm.py` – Hugging Face text generation wrapper with graceful fallback if transformers is missing
  - `canary.py` – canary document design and corpus generator
  - `pipeline.py` – end-to-end RAG pipeline (build index + query)
- `scripts/`
  - `generate_canaries.py` – materialize canary corpus under `data/canaries/`
  - `generate_benign_corpus.py` – generate a synthetic benign text corpus
  - `build_index.py` – build vector index from benign corpus + canaries
  - `query_rag.py` – CLI for querying the RAG system
  - `sanity_check.py` – check canary hit rate for benign queries
- `data/`
  - `corpus/` – place your benign `.txt` files here
  - `canaries/` – generated canary `.txt` files
  - `combined/` – optional workspace if you want to merge corpora
- `artifacts/`
  - `faiss_index.bin.npy` – saved embedding matrix (kept under this base name for compatibility)
  - `metadata.jsonl` – per-vector metadata (doc id, canary flags, text)

## Step 1 – Prepare a Benign Corpus

Option A – add your own normal documents as plain text files:

```bash
mkdir -p data/corpus
echo "This is an example product FAQ..." > data/corpus/example_faq.txt
```

Use your own internal knowledge base or any test corpus; the only requirement is
that canary markers should not appear in normal documents.

Option B – auto-generate a large synthetic benign corpus:

```bash
# Balanced 1,500-document corpus with all types (incident/changelog/FAQ/guide/design/policy)
python scripts/generate_benign_corpus.py --num-docs 1500 --profile balanced --seed 123

# More engineering-heavy corpus
python scripts/generate_benign_corpus.py --num-docs 500 --profile engineering

# More support/FAQ-heavy corpus into a different folder
python scripts/generate_benign_corpus.py --num-docs 500 --profile support --output-dir data/support_corpus
```

These commands populate the chosen directory with a few hundred reasonably long
“normal” docs across topics like FAQs, how‑tos, design notes, incident reports,
changelogs, and internal policies (depending on the selected profile). None of
these contain the canary marker tokens.

## Step 2 – Generate Canary Documents

Run:

```bash
python scripts/generate_canaries.py
```

This will create files like:

- `data/canaries/keyword_canary_00.txt`
- `data/canaries/semantic_canary_00.txt`

### Canary Design Choices

- **Keyword-dense canaries**
  - Use a rare marker token such as `ALPHA-CANARY-KEYWORD-123`.
  - Text describes a fictional, obscure protocol unrelated to any real product.
  - The marker is repeated many times to make retrieval highly sensitive to its presence.
  - Intent: detect **poisoning / boosting** where an attacker forces this token into
    documents or queries.

- **Semantically general canaries**
  - Look like ordinary internal documentation (e.g., knowledge management best practices).
  - Embed a subtle marker token such as `OMEGA-CANARY-SEMANTIC-987` once or twice.
  - Topic is generic and overlaps real docs, but the marker is artificial.
  - Intent: detect **probing / membership inference** when an adversary tries to
    induce the model to leak specific internal documents.

- **Irrelevance to benign queries**
  - Marker tokens are deliberately unnatural and should never appear in normal logs.
  - Canary topics are obscure (keyword-dense) or high-level but non-task-specific
    (semantic), so typical user questions (e.g., "reset password") should not match them.

- **Indistinguishable structure**
  - Canary documents follow the same format as real docs: paragraphs, neutral tone,
    full sentences, etc.
  - This makes it hard for an attacker to identify them purely from structure.

## Step 3 – Build the RAG Index

After you have benign docs in `data/corpus/` and have generated canaries:

```bash
python scripts/build_index.py
```

This will:

- Read all `.txt` files under `data/corpus/` and `data/canaries/`
- Chunk them into overlapping spans
- Embed chunks using the hashing-based embedder
- Build a NumPy-based index (cosine similarity)
- Save:
  - `artifacts/faiss_index.bin.npy`
  - `artifacts/metadata.jsonl`

Each metadata entry records:

- `doc_id` – original file path relative to its root
- `chunk_id` – within-document chunk index
- `is_canary` – boolean
- `canary_type` – `"keyword"`, `"semantic"`, or `"unknown"`

## Step 4 – Query the RAG System

Once the index is built:

```bash
python scripts/query_rag.py "How do I reset my password?"
```

Output:

- Generated answer from the LLM, restricted to retrieved context.
- List of retrieved chunks with:
  - similarity scores
  - document IDs
  - `canary` flag and type

This gives you a working baseline RAG system with visible canary participation.

## Step 5 – Sanity Check Canaries Under Benign Usage

Run:

```bash
python scripts/sanity_check.py
```

What it does:

- Issues a small set of hand-picked benign queries.
- For each query, retrieves top-k chunks from the index.
- Reports whether any retrieved chunk is a canary.
- Prints the final count of benign queries that surfaced canaries.

Expected behavior for a reasonably benign corpus:

- Canary hit rate should be **very low or zero** for these queries.
- If canaries frequently appear, you may:
  - Reduce their number or adjust their content.
  - Increase corpus size or diversity.
  - Use this behavior as a red flag in later detection stages.

## Customization

- Change model names, data directories, and canary tokens in `canaryrag/config.py`.
- Swap out the default `gpt2` for a different open-source causal LLM (e.g., `gpt2-medium`)
  if you have the compute and storage. If `transformers` is not installed, the
  pipeline will still run and return a clear placeholder answer.
- The current embedding model is intentionally lightweight and dependency-free; you
  can replace `EmbeddingModel` with a SentenceTransformer-based model if you prefer.
- Adjust `chunk_size` and `chunk_overlap` for your document distribution.

## What Is Implemented (Person 1 Scope)

- **Baseline RAG pipeline**
  - [x] Embedding model (lightweight hashing encoder)
  - [x] Vector store (NumPy cosine index + JSONL metadata)
  - [x] Optional open-source LLM for generation (HF `gpt2`, configurable, with graceful fallback)
- **Benign corpus**
  - [x] Generator for large synthetic corpora with multiple profiles
  - [x] Balanced 1,500-document corpus option across incident/changelog/FAQ/guide/design/policy
- **Canaries**
  - [x] Keyword-dense and semantically general canary document generators
  - [x] Canary corpus materialization script (`generate_canaries.py`)
  - [x] Structural and semantic design choices documented in this README
- **Sanity checks**
  - [x] Script to validate canary behavior under benign queries (`sanity_check.py`)
  - [x] CLI tooling to inspect retrieved chunks and canary participation (`query_rag.py`)
