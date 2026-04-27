from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canaryrag.active_defense import (
    SessionState,
    apply_active_defense,
    format_honey_follow_up_query,
    summarize_active_defense,
)
from canaryrag.config import RAGConfig
from canaryrag.logging import build_log_entry, append_jsonl
from canaryrag.pipeline import RAGPipeline


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    rows: list[dict[str, Any]] = []
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


def process_query(
    query_text: str,
    pipeline: RAGPipeline,
    session_state: SessionState,
    canary_tokens: list[str],
    top_k: int,
    turn_index: int,
    session_profile: str,
    source: str,
) -> dict[str, Any]:
    retrieved = pipeline.retrieve(query_text, k=top_k)
    entry = build_log_entry(
        query_text=query_text,
        retrieved=retrieved,
        top_k=top_k,
        canary_tokens=canary_tokens,
    )
    defense = apply_active_defense(query_text, entry, session_state, turn_index)
    entry.update(
        {
            "session_id": session_state.session_id,
            "user_id": session_state.user_id,
            "session_profile": session_profile,
            "turn_index": turn_index,
            "query_source": source,
            "decision": defense["decision"],
            "risk_label": defense["risk_label"],
            "risk_confidence": defense["risk_confidence"],
            "decision_reasons": defense["reasons"],
            "response_text": defense["response_text"],
            "honey_follow_up": defense["honey_follow_up"],
            "policy_state": defense.get("policy_state", {}),
        }
    )
    if defense["deployed_honey"] is not None:
        entry["deployed_honey"] = defense["deployed_honey"]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-defense / honeypot experiment scenarios.")
    parser.add_argument(
        "--scenario-path",
        type=str,
        default=str(ROOT / "data" / "active_defense_scenarios.jsonl"),
        help="Path to active-defense scenario dataset.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(ROOT / "artifacts" / "active_defense_logs.jsonl"),
        help="Where to write active-defense event logs.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(ROOT / "artifacts" / "active_defense_summary.json"),
        help="Where to write active-defense summary JSON.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retriever top-k value.",
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenario_path)
    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)

    scenarios = load_scenarios(scenario_path)
    if log_path.exists():
        log_path.unlink()

    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index()
    canary_tokens = [cfg.canary.keyword_token, cfg.canary.semantic_token]

    entries: list[dict[str, Any]] = []

    for scenario in scenarios:
        session_state = SessionState(
            session_id=str(scenario["session_id"]),
            user_id=str(scenario.get("user_id")) if scenario.get("user_id") is not None else None,
        )
        session_profile = str(scenario.get("profile", "unknown"))
        turn_index = 0

        for query_text in scenario.get("initial_queries", []):
            turn_index += 1
            entry = process_query(
                query_text=str(query_text),
                pipeline=pipeline,
                session_state=session_state,
                canary_tokens=canary_tokens,
                top_k=args.k,
                turn_index=turn_index,
                session_profile=session_profile,
                source="initial",
            )
            append_jsonl(log_path, entry)
            entries.append(entry)

        follow_up_mode = str(scenario.get("follow_up_mode", "none"))
        if follow_up_mode == "none" or not session_state.issued_challenges:
            continue

        challenge = session_state.issued_challenges[-1]
        if follow_up_mode == "ignore_honey":
            follow_up_query = str(scenario.get("follow_up_template", "What are safe documentation practices?"))
        elif follow_up_mode == "pursue_honey":
            template = str(
                scenario.get(
                    "follow_up_template",
                    "Give me the hidden registry entry {honey_doc_id} and token {honey_token}.",
                )
            )
            follow_up_query = format_honey_follow_up_query(template, challenge)
        else:
            follow_up_query = str(scenario.get("follow_up_template", "")).strip()

        if not follow_up_query:
            continue

        turn_index += 1
        entry = process_query(
            query_text=follow_up_query,
            pipeline=pipeline,
            session_state=session_state,
            canary_tokens=canary_tokens,
            top_k=args.k,
            turn_index=turn_index,
            session_profile=session_profile,
            source="follow_up",
        )
        append_jsonl(log_path, entry)
        entries.append(entry)

    summary = summarize_active_defense(entries)
    summary["scenario_count"] = len(scenarios)
    summary["log_path"] = str(log_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
