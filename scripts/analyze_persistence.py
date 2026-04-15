from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
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


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class SessionSummary:
    session_id: str
    user_id: str | None
    total_queries: int
    flagged_queries: int
    canary_seen: bool
    marker_seen: bool
    persistent_follow_up: bool


def summarize_sessions(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Group by session_id (fallback to user_id, then query_id).
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        sid = str(e.get("session_id") or "") or str(e.get("user_id") or "") or str(e.get("query_id") or "")
        sessions[sid].append(e)

    summaries: List[SessionSummary] = []

    for sid, sess_entries in sessions.items():
        # Sort by timestamp if available.
        sess_entries_sorted = sorted(
            sess_entries,
            key=lambda x: parse_ts(str(x.get("timestamp"))) or datetime.min,
        )

        user_id = sess_entries_sorted[0].get("user_id")
        total = len(sess_entries_sorted)
        canary_seen = any(int(e.get("canary_count", 0)) > 0 for e in sess_entries_sorted)
        marker_seen = any(int(e.get("marker_hits", 0)) > 0 for e in sess_entries_sorted)

        # "Flagged" queries: heuristic based on retrieval signals.
        flagged_indices: List[int] = []
        for idx, e in enumerate(sess_entries_sorted):
            if int(e.get("canary_count", 0)) > 0 or int(e.get("marker_hits", 0)) > 0:
                flagged_indices.append(idx)

        flagged_queries = len(flagged_indices)

        persistent_follow_up = False
        if flagged_indices:
            first_flag = flagged_indices[0]
            for e in sess_entries_sorted[first_flag + 1 :]:
                if int(e.get("canary_count", 0)) > 0 or int(e.get("marker_hits", 0)) > 0:
                    persistent_follow_up = True
                    break

        summaries.append(
            SessionSummary(
                session_id=sid,
                user_id=str(user_id) if user_id is not None else None,
                total_queries=total,
                flagged_queries=flagged_queries,
                canary_seen=canary_seen,
                marker_seen=marker_seen,
                persistent_follow_up=persistent_follow_up,
            )
        )

    total_sessions = len(summaries)
    benign_sessions = 0
    curious_sessions = 0
    persistent_sessions = 0

    for s in summaries:
        if s.flagged_queries == 0:
            benign_sessions += 1
        elif s.flagged_queries == 1 and not s.persistent_follow_up:
            curious_sessions += 1
        else:
            persistent_sessions += 1

    return {
        "total_sessions": total_sessions,
        "session_categories": {
            "benign": benign_sessions,
            "curious": curious_sessions,
            "persistent": persistent_sessions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze session-level persistence from retrieval logs.")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(ROOT / "artifacts" / "retrieval_logs.jsonl"),
        help="Path to retrieval log JSONL (with session_id/user_id if available).",
    )
    args = parser.parse_args()

    entries = load_jsonl(Path(args.log_path))
    summary = summarize_sessions(entries)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

