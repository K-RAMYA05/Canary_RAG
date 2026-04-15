from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_detection import classify_entry as rb_classify  # type: ignore[import]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def to_binary_label(label: str) -> str:
    """
    Map multi-class detector labels to a binary intent label.
    - 'benign' stays 'benign'
    - 'probing' and 'poisoning' are treated as 'probing' (attack) for intent detection.
    """
    label = label.lower()
    if label == "benign":
        return "benign"
    return "probing"


def compute_confusion(truth: List[str], preds: List[str]) -> Dict[str, float]:
    if len(truth) != len(preds):
        raise ValueError("truth and preds must have same length")

    tp = fp = tn = fn = 0
    for y_true, y_pred in zip(truth, preds):
        is_attack = y_true == "probing"
        pred_attack = y_pred == "probing"
        if is_attack and pred_attack:
            tp += 1
        elif not is_attack and pred_attack:
            fp += 1
        elif not is_attack and not pred_attack:
            tn += 1
        elif is_attack and not pred_attack:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fpr, 3),
    }


def evaluate(
    retrieval_log_path: Path,
    gatekeeper_log_path: Path | None,
) -> Dict[str, Any]:
    retrieval_entries = load_jsonl(retrieval_log_path)
    # Only keep entries that have ground-truth labels from the query dataset.
    labeled_entries = [e for e in retrieval_entries if "label" in e and e.get("label") is not None]

    if not labeled_entries:
        raise ValueError("No labeled entries found in retrieval logs (expected 'label' field).")

    y_true = [str(e["label"]).lower() for e in labeled_entries]

    # Rule-based detector evaluation (computed on the fly).
    rb_preds_binary: List[str] = []
    rb_raw_labels: List[str] = []
    for e in labeled_entries:
        out = rb_classify(e)
        raw_label = str(out.get("label", "benign")).lower()
        rb_raw_labels.append(raw_label)
        rb_preds_binary.append(to_binary_label(raw_label))

    rb_metrics = compute_confusion(y_true, rb_preds_binary)

    results: Dict[str, Any] = {
        "num_labeled_queries": len(labeled_entries),
        "rule_based": {
            "metrics": rb_metrics,
            "labels_used": {
                "benign": rb_raw_labels.count("benign"),
                "probing": rb_raw_labels.count("probing"),
                "poisoning": rb_raw_labels.count("poisoning"),
            },
        },
    }

    # Optional LLM gatekeeper evaluation, if outputs are provided.
    if gatekeeper_log_path is not None and gatekeeper_log_path.exists():
        gate_entries = load_jsonl(gatekeeper_log_path)
        by_qid: Dict[str, Dict[str, Any]] = {}
        for g in gate_entries:
            qid = str(g.get("query_id") or "")
            if qid:
                by_qid[qid] = g

        gk_preds_binary: List[str] = []
        matched_truth: List[str] = []

        for e in labeled_entries:
            qid = str(e.get("query_id") or "")
            if not qid:
                continue
            if qid not in by_qid:
                continue
            g = by_qid[qid]
            label = str(g.get("final_label", "benign")).lower()
            gk_preds_binary.append(to_binary_label(label))
            matched_truth.append(str(e["label"]).lower())

        if matched_truth:
            gk_metrics = compute_confusion(matched_truth, gk_preds_binary)
            results["llm_gatekeeper"] = {
                "metrics": gk_metrics,
                "num_evaluated": len(matched_truth),
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection and attribution over labeled query logs.")
    parser.add_argument(
        "--retrieval-log",
        type=str,
        default=str(ROOT / "artifacts" / "retrieval_logs.jsonl"),
        help="Path to retrieval log JSONL (with 'label' field).",
    )
    parser.add_argument(
        "--gatekeeper-log",
        type=str,
        default=str(ROOT / "artifacts" / "final_detections.jsonl"),
        help="Optional path to LLM gatekeeper output JSONL.",
    )
    args = parser.parse_args()

    retrieval_log_path = Path(args.retrieval_log)
    gatekeeper_log_path = Path(args.gatekeeper_log) if args.gatekeeper_log else None

    summary = evaluate(retrieval_log_path, gatekeeper_log_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

