from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.config import RAGConfig
from canaryrag.logging import build_log_entry
from canaryrag.pipeline import RAGPipeline, read_text_files
from scripts.rule_based_detection import classify_entry


HEADER_RE = re.compile(
    r"^(?P<label>.+?) for (?P<product>[A-Za-z0-9_-]+) in (?P<region>[A-Za-z0-9_-]+) maintained by (?P<team>.+?)\.$"
)
SEVERITY_RE = re.compile(r"^Severity:\s*(?P<value>.+?)\.$")
AREA_RE = re.compile(r"^Impacted area:\s*(?P<value>.+?)\.$")


QUERY_TEMPLATES = {
    "faq": [
        "How do I get support guidance for {product} in {region}?",
        "Where can I find the user-facing FAQ for {product} maintained by {team}?",
    ],
    "guide": [
        "What is the internal guide for working on {product} in {region}?",
        "How should {team} handle routine workflows for {product}?",
    ],
    "design": [
        "Explain the architecture guidance for {product} maintained by {team}.",
        "What design considerations apply to {product} in {region}?",
    ],
    "incident": [
        "Summarize the {severity} incident affecting {area} for {product}.",
        "What happened in the {product} incident involving {area}?",
    ],
    "changelog": [
        "What changed recently for {product} in {region}?",
        "Summarize the latest release updates for {product}.",
    ],
    "policy": [
        "What policy guidance applies to {team} working on {product}?",
        "Summarize the internal policy for {product} in {region}.",
    ],
}


def infer_doc_type(doc_id: str) -> str:
    return doc_id.split("_", 1)[0].lower()


def parse_doc_fields(text: str) -> dict[str, str]:
    fields = {
        "product": "Atlas",
        "region": "us-east",
        "team": "Core Platform",
        "severity": "medium",
        "area": "public API",
    }

    for line in text.splitlines()[:12]:
        line = line.strip()
        if not line:
            continue

        match = HEADER_RE.match(line)
        if match:
            fields["product"] = match.group("product")
            fields["region"] = match.group("region")
            fields["team"] = match.group("team")
            continue

        severity_match = SEVERITY_RE.match(line)
        if severity_match:
            fields["severity"] = severity_match.group("value")
            continue

        area_match = AREA_RE.match(line)
        if area_match:
            fields["area"] = area_match.group("value")

    return fields


def make_benign_queries(docs: list[tuple[str, str]], queries_per_doc: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for idx, (doc_id, text) in enumerate(docs):
        doc_type = infer_doc_type(doc_id)
        templates = QUERY_TEMPLATES.get(doc_type)
        if not templates:
            continue

        fields = parse_doc_fields(text)
        chosen = rng.sample(templates, k=min(queries_per_doc, len(templates)))
        for q_idx, template in enumerate(chosen):
            query_text = template.format(**fields)
            rows.append(
                {
                    "query_id": f"benign-{idx:05d}-{q_idx}",
                    "query_text": query_text,
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                }
            )

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    canary_hits = sum(1 for row in rows if int(row.get("canary_count", 0)) > 0)
    top1_canary_hits = sum(1 for row in rows if bool(row.get("top1_is_canary")))
    false_positives = sum(1 for row in rows if str(row.get("predicted_label")) != "benign")

    by_doc_type: dict[str, dict[str, int]] = {}
    for row in rows:
        doc_type = str(row.get("doc_type", "unknown"))
        stats = by_doc_type.setdefault(
            doc_type,
            {"total": 0, "canary_hits": 0, "top1_canary_hits": 0, "false_positives": 0},
        )
        stats["total"] += 1
        if int(row.get("canary_count", 0)) > 0:
            stats["canary_hits"] += 1
        if bool(row.get("top1_is_canary")):
            stats["top1_canary_hits"] += 1
        if str(row.get("predicted_label")) != "benign":
            stats["false_positives"] += 1

    def rates(stats: dict[str, int]) -> dict[str, float]:
        total_local = stats["total"]
        return {
            "total": total_local,
            "canary_hit_rate": stats["canary_hits"] / total_local if total_local else 0.0,
            "top1_canary_rate": stats["top1_canary_hits"] / total_local if total_local else 0.0,
            "false_positive_rate": stats["false_positives"] / total_local if total_local else 0.0,
        }

    return {
        "total_queries": total,
        "overall": {
            "canary_hit_rate": canary_hits / total if total else 0.0,
            "top1_canary_rate": top1_canary_hits / total if total else 0.0,
            "false_positive_rate": false_positives / total if total else 0.0,
        },
        "by_doc_type": {key: rates(value) for key, value in by_doc_type.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benign-search false positives on a larger synthetic benign query set.")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=str(ROOT / "data" / "corpus"),
        help="Directory containing benign corpus text files.",
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=300,
        help="Maximum number of benign documents to sample.",
    )
    parser.add_argument(
        "--queries-per-doc",
        type=int,
        default=2,
        help="How many benign queries to generate per sampled document.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for document sampling.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retriever top-k.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(ROOT / "artifacts" / "benign_traffic_summary.json"),
        help="Where to write the benign-traffic summary JSON.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(ROOT / "artifacts" / "benign_traffic_logs.jsonl"),
        help="Where to write per-query benign-traffic logs.",
    )
    args = parser.parse_args()

    docs = read_text_files(Path(args.corpus_dir))
    rng = random.Random(args.seed)
    if len(docs) > args.sample_docs:
        docs = rng.sample(docs, k=args.sample_docs)

    generated = make_benign_queries(docs, queries_per_doc=args.queries_per_doc, seed=args.seed)

    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()
    canary_tokens = [cfg.canary.keyword_token, cfg.canary.semantic_token]

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    evaluated: list[dict[str, Any]] = []
    for row in generated:
        retrieved = pipeline.retrieve(str(row["query_text"]), k=args.k)
        entry = build_log_entry(
            query_text=str(row["query_text"]),
            retrieved=retrieved,
            top_k=args.k,
            canary_tokens=canary_tokens,
            query_id=str(row["query_id"]),
        )
        prediction = classify_entry(entry)
        combined = {
            **row,
            **entry,
            "predicted_label": prediction["label"],
            "prediction_confidence": prediction["confidence"],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(combined, ensure_ascii=False) + "\n")
        evaluated.append(combined)

    summary = summarize(evaluated)
    summary["sampled_documents"] = len(docs)
    summary["generated_queries"] = len(generated)
    summary["log_path"] = str(log_path)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
