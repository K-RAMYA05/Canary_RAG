# CanaryRAG – Baseline RAG + Canary Design

This repository implements **Person 1**'s scope from the project description:

- Baseline RAG pipeline (embeddings + vector DB + open-source LLM)
- Synthetic canary document design and insertion
- Sanity checks for retrieval behavior under benign usage
- Large synthetic benign corpus generation

## Overview

- **Embedding model**: lightweight hashing-based embedder with stable hashing and dedicated canary dimensions
- **Vector database**: NumPy-based cosine-similarity index with JSONL metadata
- **LLM for generation**: Hugging Face causal LM, configurable in `canaryrag/config.py`
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

> Note: the first answer-generation run may download the configured Hugging Face
> causal LM, which requires network access. Index building and retrieval-only
> workflows do not require model downloads.

## Project Layout

- `canaryrag/`
  - `config.py` – central configuration for models, data paths, and canary tokens
  - `embeddings.py` – simple hashing-based embedding model with stable token hashing
  - `vectorstore.py` – in-memory cosine-similarity vector store + JSONL metadata
  - `llm.py` – Hugging Face text generation wrapper with graceful fallback if transformers is missing
  - `canary.py` – canary document design and corpus generator
  - `pipeline.py` – end-to-end RAG pipeline (build index + query)
  - `active_defense.py` – session-aware honey-text deployment and follow-up escalation logic
- `scripts/`
  - `generate_canaries.py` – materialize canary corpus under `data/canaries/`
  - `generate_benign_corpus.py` – generate a synthetic benign text corpus
  - `build_index.py` – build vector index from benign corpus + canaries
  - `query_rag.py` – CLI for querying the RAG system
  - `sanity_check.py` – check canary hit rate for benign queries
  - `simulate_poisoning.py` – create a poisoned corpus by copying the clean corpus and injecting malicious documents
  - `run_poisoning_experiment.py` – build clean vs poisoned indexes and compare retrieval behavior
  - `run_active_defense_experiment.py` – run session-based honey-text / active-defense scenarios
- `data/`
  - `corpus/` – place your benign `.txt` files here
  - `canaries/` – generated canary `.txt` files
  - `combined/` – optional workspace if you want to merge corpora
  - `active_defense_scenarios.jsonl` – scenario dataset for active-defense evaluation
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
- By default, retrieved canary chunks are excluded from the answer-generation prompt
  and used only as detection signals.
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

## Step 6 – Simulate Corpus Poisoning

To model actual index poisoning rather than only poisoning-flavored prompts:

```bash
python scripts/run_poisoning_experiment.py
```

What it does:

- Copies `data/corpus/` into `data/poisoned_corpus/`
- Injects synthetic poisoned documents that resemble normal content while carrying canary-aligned markers
- Builds:
  - a clean comparison index
  - a poisoned comparison index
- Runs the labeled query set against both
- Saves side-by-side logs and a summary JSON under `artifacts/`

Useful outputs:

- `artifacts/poisoning_clean_retrieval_logs.jsonl`
- `artifacts/poisoning_retrieval_logs.jsonl`
- `artifacts/poisoning_experiment_summary.json`

## Step 7 – Run Active Defense / Honeypot Evaluation

To evaluate session-level honey-text deployment and bait-follow-up escalation:

```bash
python scripts/run_active_defense_experiment.py
```

What it does:

- Loads session scenarios from `data/active_defense_scenarios.jsonl`
- Runs normal retrieval for each query
- Deploys a session-specific honey registry reference when a query is suspicious
- Distinguishes:
  - benign sessions
  - curious sessions that do not pursue the bait
  - persistent sessions that explicitly follow the bait
- Escalates sessions that reference the issued honey token or honey registry entry

Useful outputs:

- `artifacts/active_defense_logs.jsonl`
- `artifacts/active_defense_summary.json`

## Step 8 – Run Practicality Study

To quantify how canary count affects retrieval quality and overhead:

```bash
python scripts/run_practicality_study.py
```

What it does:

- Sweeps several total canary counts
- Builds isolated indexes for each setting
- Measures:
  - canary hit rates on the labeled query set
  - index size
  - metadata size
  - build time
  - batch query time
  - chunk growth as canaries increase

Useful outputs:

- `artifacts/practicality_study_summary.json`
- per-setting index and metadata files under `artifacts/practicality/`

## Customization

- Change model names, data directories, generation behavior, and canary tokens in `canaryrag/config.py`.
- Swap out the default causal LM for a different open-source model if you have
  the compute and storage. If `transformers` is not installed, the pipeline will
  still run and return a clear placeholder answer.
- The current embedding model is intentionally lightweight and dependency-free; you
  can replace `EmbeddingModel` with a SentenceTransformer-based model if you prefer.
- Adjust `chunk_size` and `chunk_overlap` for your document distribution.
- If you explicitly want canaries included in answer-generation context for
  debugging, set `include_canaries_in_generation=True` in `DataConfig`.

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



# CanaryRAG – Detection + Attribution (Person 2)

This repository implements **Person 2’s scope** from the project description:

* Retrieval logging for all queries
* Rule-based detection using canary signals
* Attribution logic (benign vs probing)
* Lightweight LLM gatekeeper for decision refinement
* False-positive reduction through semantic validation



# Overview

The detection layer sits **on top of the baseline RAG system** and converts retrieval signals into **security decisions**.

Pipeline:

```
Query → Retrieval → Logging → Rule-based Detection → LLM Gatekeeper → Final Decision
```


# Core Components

### Retrieval Logging

All queries are logged in structured JSONL format with:

* query_id
* timestamp
* query_text
* top_k
* retrieved chunks:

  * rank
  * score
  * doc_id
  * chunk_id
  * is_canary
  * canary_type

Derived signals:

* canary_count
* top1_is_canary
* best_canary_rank
* repetition_count
* marker_hits

Purpose:

* provide traceability
* enable downstream detection and analysis



### Rule-Based Detection

A lightweight detection system using retrieval signals.

Key rules:

* **Canary presence in query**

  ```
  marker_hits > 0 → probing
  ```

* **Canary presence in retrieval**

  ```
  canary_count > 0 → suspicious
  ```

* **Rank-based signals**

  * canary in top-1
  * canary in top-k

* **Repetition signal**

  * repeated tokens in query

Purpose:

* fast first-stage detection
* interpretable decisions



### Attribution Logic

Each query is classified as:

* **benign**
* **probing**

Examples:

| Query                         | Label   |
| ----------------------------- | ------- |
| "How do I reset my password?" | benign  |
| "ALPHA-CANARY-KEYWORD-123"    | probing |
| "OMEGA-CANARY-SEMANTIC-987"   | probing |



### LLM Gatekeeper (Lightweight)

A second-stage validator applied only to flagged queries.

Input:

* query
* rule-based decision
* top retrieved chunks

Output:

```json
{
  "final_label": "benign | probing",
  "confidence": 0.0-1.0,
  "reason": "...",
  "should_escalate": true | false
}
```

Purpose:

* refine rule-based decisions
* incorporate semantic understanding
* reduce false positives



### False Positive Reduction

Handled by the gatekeeper.

Example:

```
Query: "Explain ALPHA-CANARY-KEYWORD-123 usage"

Rule-based → probing
LLM → benign
```

This allows:

* educational queries → benign
* direct probing → probing


# Scripts

### scripts/query_with_retrieval_log.py

* Runs query
* Logs retrieval results
* Generates signals

### scripts/llm_gatekeeper.py

* Applies rule-based detection
* Calls LLM for flagged queries
* Produces final decision



# Step-by-Step Usage

## Step 1 – Run a Query with Logging

```bash
export PYTHONHASHSEED=0
python scripts/query_with_retrieval_log.py "How do I reset my password?"
```

---

## Step 2 – Run Detection + Gatekeeper

```bash
python scripts/llm_gatekeeper.py --latest
```


# Example Outputs

## Benign Query

```json
{
  "query_text": "How do I reset my password?",
  "final_label": "benign",
  "confidence": 0.97
}
```


## Strong Probing

```json
{
  "query_text": "ALPHA-CANARY-KEYWORD-123",
  "final_label": "probing",
  "confidence": 0.99
}
```



## False Positive Reduced

```json
{
  "query_text": "Explain ALPHA-CANARY-KEYWORD-123 usage",
  "final_label": "benign",
  "confidence": 0.95
}
```



# Design Choices

### Two-Stage Detection

* Stage 1: Rule-based (fast, deterministic)
* Stage 2: LLM gatekeeper (semantic reasoning)

---

### Canary Signals

* keyword canary → strong signal
* semantic canary → weak, subtle signal



### Lightweight Models

* small LLM used for gatekeeper
* avoids heavy compute requirements



# What Is Implemented (Person 2 Scope)

### Detection Pipeline

* End-to-end flow from query to decision
* Logging → rules → LLM → final output

### Attribution Logic

* benign vs probing classification
* confidence scoring
* explanation generation

### LLM Gatekeeper

* applied only to flagged queries
* reduces false positives
* structured JSON output

### Signal-Based Detection

* canary presence
* rank thresholds
* repetition detection
