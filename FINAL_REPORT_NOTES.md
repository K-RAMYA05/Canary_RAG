# CanaryRAG Final Report Notes

## Final Positioning
CanaryRAG is a retrieval-layer defense for RAG systems that combines synthetic canary documents, retrieval-time monitoring, poisoning analysis, and honeypot-style active defense. The system is designed to detect suspicious retrieval behavior without requiring heavy inspection of every user query, and to escalate sessions that pursue controlled bait references.

## What Is Implemented

### 1. Baseline RAG Pipeline
- Lightweight hashing-based embedding model with stable hashing
- NumPy cosine-similarity vector store with JSONL metadata
- Chunking, indexing, retrieval, and optional answer generation
- Canary chunks filtered from answer-generation context by default

### 2. Canary Document Design
- Keyword-dense canaries
  - optimized for direct token probes, repeated markers, and explicit retrieval targeting
- Semantically general canaries
  - designed to resemble normal internal text while carrying subtle machine-detectable markers
- Canary corpus generator and materialization scripts

### 3. Retrieval Logging and Detection
- Structured retrieval logs with:
  - query text
  - retrieved ranks and scores
  - canary counts
  - marker hits
  - repetition counts
- Rule-based detector for benign vs suspicious / probing behavior
- Optional LLM gatekeeper prototype for second-stage review

### 4. Persistence Analysis
- Session-level categorization into:
  - benign
  - curious
  - persistent
- Tracking of repeated suspicious follow-up behavior across turns

### 5. Poisoning Experiment
- Simulated poisoning by injecting malicious documents into a copied corpus
- Clean vs poisoned index comparison
- Measurement of how poisoned documents displace canary retrieval and change top-1 results

### 6. Active Defense / Honeypot Mechanism
- Session-specific honey-token and honey-registry generation
- Controlled bait deployment for suspicious sessions
- Follow-up detection when users explicitly pursue the bait
- Automatic escalation / block behavior for persistent bait pursuit

### 7. Practicality Study
- Canary-count sweep across multiple settings
- Measurement of:
  - storage growth
  - chunk growth
  - build time
  - query time
  - benign and probing canary hit rates

## Core Results

### Clean Retrieval / Detection
- Rule-based detector:
  - accuracy: `1.0`
  - precision: `1.0`
  - recall: `1.0`
  - f1: `1.0`
  - false-positive rate: `0.0`
- Session persistence summary:
  - benign sessions: `10`
  - curious sessions: `1`
  - persistent sessions: `6`

Clean retrieval summary:
- benign canary hit rate: `0.0`
- probing canary hit rate: `1.0`
- probing semantic canary hit rate: `1.0`

### Poisoning Experiment
- Clean probing canary hit rate: `1.0`
- Poisoned probing canary hit rate: `0.75`
- Probing poisoned-doc hit rate: `0.25`
- Probing top-1 poisoned-doc rate: `0.25`
- Benign top-1 poisoned-doc rate after refinement: `0.0`

Interpretation:
- Poisoned documents can suppress canary retrieval signals by competing for top-ranked retrieval positions.
- After refinement, the poisoned semantic documents no longer capture benign top-1 queries in the labeled set.
- The clean system now catches both explicit marker probes and the semantic probing slice in the current evaluation set.

### Active Defense / Honeypot
- Total turns evaluated: `18`
- Honey deployments: `6`
- Escalations: `4`
- Benign false escalations: `0`

By profile:
- benign:
  - sessions: `3`
  - honey sessions: `0`
  - escalated sessions: `0`
- curious:
  - sessions: `2`
  - honey sessions: `2`
  - escalated sessions: `0`
- persistent:
  - sessions: `4`
  - honey sessions: `4`
  - escalated sessions: `4`

Interpretation:
- The active-defense mechanism successfully separates curiosity from adversarial persistence in the controlled session evaluation.

### Practicality Study
Counts tested:
- `4`, `10`, `20`, `50` total canaries

Observed behavior:
- benign canary hit rate remained `0.0`
- probing canary hit rate remained `1.0`
- query latency remained about `0.0011s` to `0.0013s` per query
- storage overhead grew modestly between the smallest and largest tested settings

Interpretation:
- Moderate increases in canary volume do not materially change retrieval behavior or local runtime overhead in the current prototype.

## Main Contributions
- Demonstrates that canary retrieval behavior can serve as an interpretable security signal at the retrieval layer.
- Shows that poisoning can suppress dedicated canary retrieval, motivating explicit poisoning analysis rather than probe-only evaluation.
- Implements a concrete honeypot-style active-defense layer with session-specific bait and escalation.
- Provides practicality evidence that the approach remains lightweight under moderate canary scaling.

## Limitations
- The evaluation is based on synthetic corpora and controlled synthetic query/session sets.
- The vector store is a lightweight NumPy implementation rather than a production-scale deployment.
- The LLM gatekeeper exists but is less central than the rule-based and active-defense components.
- The current semantic coverage is stronger than before, but it is still achieved through an intentionally engineered lightweight embedding space rather than a fully general semantic encoder.
- The active-defense evaluation demonstrates controlled behavior, not live deployment evidence.

## Honest Final Claim
We implemented a full prototype of CanaryRAG that includes passive canary-based retrieval monitoring, poisoning analysis, session-level persistence tracking, and honeypot-style active defense. The strongest evidence is on controlled probing, poisoning, and bait-pursuit scenarios in a synthetic evaluation environment.

## Suggested Presentation Flow
1. Explain the RAG security problem:
   probing, membership inference, and poisoning.
2. Show the baseline idea:
   canaries as retrieval-layer tripwires.
3. Show the passive results:
   rule-based detection and persistence analysis.
4. Show the poisoning experiment:
   clean vs poisoned retrieval behavior.
5. Show the active-defense mechanism:
   honey-token deployment and escalation.
6. Show the practicality study:
   stable detection and low overhead as canaries scale.
7. End with limitations and future work.

## Suggested Future Work
- Improve broader semantic-probing coverage
- Evaluate on more realistic corpora and user sessions
- Test with production-style vector databases and deployment pipelines
- Expand the active-defense policy beyond direct bait-reference follow-up
