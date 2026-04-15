from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .vectorstore import VectorMetadata


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+", text.lower())


def count_repetition(text: str) -> int:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    return sum(count - 1 for count in counts.values() if count > 1)


def count_marker_hits(text: str, markers: list[str]) -> int:
    q = text.lower()
    return sum(q.count(marker.lower()) for marker in markers)


def build_log_entry(
    query_text: str,
    retrieved: list[dict[str, Any]],
    top_k: int,
    canary_tokens: list[str],
    query_id: str | None = None,
) -> dict[str, Any]:
    qid = query_id or uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()

    retrieved_items: list[dict[str, Any]] = []
    for rank, item in enumerate(retrieved, start=1):
        meta: VectorMetadata = item["metadata"]
        retrieved_items.append(
            {
                "rank": rank,
                "score": float(item["score"]),
                "doc_id": meta.doc_id,
                "chunk_id": int(meta.chunk_id),
                "is_canary": bool(meta.is_canary),
                "canary_type": meta.canary_type,
            }
        )

    canary_items = [x for x in retrieved_items if x["is_canary"]]
    canary_count = len(canary_items)
    top1_is_canary = bool(retrieved_items and retrieved_items[0]["is_canary"])

    best_canary_rank = canary_items[0]["rank"] if canary_items else None
    score_gap_top1_vs_top_canary = None
    if canary_items and retrieved_items:
        score_gap_top1_vs_top_canary = retrieved_items[0]["score"] - canary_items[0]["score"]

    entry = {
        "schema_version": 1,
        "query_id": qid,
        "timestamp": timestamp,
        "query_text": query_text,
        "top_k": top_k,
        "retrieved": retrieved_items,
        "canary_count": canary_count,
        "top1_is_canary": top1_is_canary,
        "best_canary_rank": best_canary_rank,
        "score_gap_top1_vs_top_canary": score_gap_top1_vs_top_canary,
        "repetition_count": count_repetition(query_text),
        "marker_hits": count_marker_hits(query_text, canary_tokens),
    }
    return entry


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

