from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


HONEY_PREFIX = "SIGMA-HONEY"
HONEY_DOC_PREFIX = "TRACE-REGISTRY"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+", text.lower())


@dataclass
class HoneyChallenge:
    challenge_id: str
    honey_token: str
    honey_doc_id: str
    issued_for_query: str
    reason: str
    challenge_text: str


@dataclass
class SessionState:
    session_id: str
    user_id: str | None = None
    issued_challenges: list[HoneyChallenge] = field(default_factory=list)
    escalated: bool = False
    suspicious_turns: int = 0
    marker_turns: int = 0
    unexpected_canary_turns: int = 0
    monitored_turns: int = 0


def classify_retrieval_risk(entry: dict[str, Any]) -> dict[str, Any]:
    marker_hits = int(entry.get("marker_hits", 0) or 0)
    canary_count = int(entry.get("canary_count", 0) or 0)
    repetition_count = int(entry.get("repetition_count", 0) or 0)
    top1_is_canary = bool(entry.get("top1_is_canary"))

    reasons: list[str] = []

    if marker_hits > 0:
        label = "probing"
        reasons.append("marker_hit")
        if canary_count > 0:
            reasons.append("marker_aligned_with_canary_retrieval")
        if top1_is_canary:
            reasons.append("top1_canary")
        confidence = 0.84 + min(marker_hits, 3) * 0.04
        if canary_count > 0:
            confidence += 0.06
        if top1_is_canary:
            confidence += 0.04
        confidence = min(confidence, 0.99)
    elif canary_count > 0:
        label = "suspicious"
        reasons.append("unexpected_canary_retrieval")
        if top1_is_canary:
            reasons.append("top1_canary")
        confidence = 0.76 + min(canary_count, 4) * 0.04
        if top1_is_canary:
            confidence += 0.05
        confidence = min(confidence, 0.95)
    elif repetition_count >= 2:
        label = "suspicious"
        reasons.append("repetition_signal")
        confidence = 0.67
    else:
        label = "benign"
        reasons.append("no_retrieval_risk_signal")
        confidence = 0.97

    should_deploy_honey = label != "benign"

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "should_deploy_honey": should_deploy_honey,
    }


def _infer_bait_context(entry: dict[str, Any], query_text: str) -> str:
    lowered_query = query_text.lower()
    retrieved = entry.get("retrieved", [])

    for item in retrieved:
        doc_id = str(item.get("doc_id", "")).lower()
        if doc_id.startswith("incident_"):
            return "incident"
        if doc_id.startswith("policy_"):
            return "policy"
        if doc_id.startswith("changelog_"):
            return "changelog"
        if doc_id.startswith("design_"):
            return "design"
        if doc_id.startswith("guide_"):
            return "guide"
        if doc_id.startswith("faq_"):
            return "faq"

    if "incident" in lowered_query or "outage" in lowered_query:
        return "incident"
    if "policy" in lowered_query or "compliance" in lowered_query:
        return "policy"
    if "release" in lowered_query or "changelog" in lowered_query:
        return "changelog"
    if "design" in lowered_query or "architecture" in lowered_query:
        return "design"
    if "guide" in lowered_query or "runbook" in lowered_query:
        return "guide"
    return "generic"


def _build_challenge_text(context: str, honey_doc_id: str, honey_token: str) -> str:
    prompt_by_context = {
        "incident": "the isolated incident validation appendix",
        "policy": "the restricted policy validation appendix",
        "changelog": "the isolated release validation appendix",
        "design": "the restricted design audit appendix",
        "guide": "the isolated migration runbook appendix",
        "faq": "the restricted support validation appendix",
        "generic": "the isolated validation appendix",
    }
    target = prompt_by_context.get(context, prompt_by_context["generic"])
    return (
        "Access to the requested material is restricted. For controlled validation, refer to "
        f"{target} `{honey_doc_id}` and validation token `{honey_token}`. "
        "If this request is part of legitimate security review, cite the appendix identifier in your next message."
    )


def build_honey_challenge(
    session_id: str,
    query_text: str,
    risk: dict[str, Any],
    turn_index: int,
    retrieval_entry: dict[str, Any],
) -> HoneyChallenge:
    seed = f"{session_id}|{turn_index}|{query_text}".encode("utf-8")
    digest = hashlib.blake2b(seed, digest_size=8).hexdigest().upper()
    honey_token = f"{HONEY_PREFIX}-{digest[:8]}"
    honey_doc_id = f"{HONEY_DOC_PREFIX}-{digest[8:12]}"
    reason = ",".join(risk.get("reasons", []))
    context = _infer_bait_context(retrieval_entry, query_text)
    challenge_text = _build_challenge_text(context, honey_doc_id, honey_token)
    return HoneyChallenge(
        challenge_id=f"{session_id}-{turn_index}",
        honey_token=honey_token,
        honey_doc_id=honey_doc_id,
        issued_for_query=query_text,
        reason=reason,
        challenge_text=challenge_text,
    )


def detect_honey_follow_up(query_text: str, challenges: list[HoneyChallenge]) -> dict[str, Any]:
    lowered = query_text.lower()
    for challenge in challenges:
        if challenge.honey_token.lower() in lowered or challenge.honey_doc_id.lower() in lowered:
            return {
                "matched": True,
                "challenge_id": challenge.challenge_id,
                "match_type": "direct_reference",
            }
    return {
        "matched": False,
        "challenge_id": None,
        "match_type": None,
        }


def _monitor_response(risk: dict[str, Any]) -> str:
    if risk["label"] == "suspicious":
        return (
            "This request produced unusual retrieval activity. The session will remain on the normal response path "
            "while additional context is gathered."
        )
    return "Normal response path allowed."


def apply_active_defense(
    query_text: str,
    retrieval_entry: dict[str, Any],
    session_state: SessionState,
    turn_index: int,
) -> dict[str, Any]:
    honey_follow_up = detect_honey_follow_up(query_text, session_state.issued_challenges)
    if honey_follow_up["matched"]:
        session_state.escalated = True
        return {
            "decision": "block_and_escalate",
            "risk_label": "persistent_adversary",
            "risk_confidence": 0.995,
            "reasons": ["honey_token_follow_up"],
            "deployed_honey": None,
            "honey_follow_up": honey_follow_up,
            "response_text": (
                "This session has been escalated due to follow-up on an isolated validation reference. "
                "No additional internal marker information will be provided."
            ),
        }

    risk = classify_retrieval_risk(retrieval_entry)
    marker_hits = int(retrieval_entry.get("marker_hits", 0) or 0)
    canary_count = int(retrieval_entry.get("canary_count", 0) or 0)
    repetition_count = int(retrieval_entry.get("repetition_count", 0) or 0)
    top1_is_canary = bool(retrieval_entry.get("top1_is_canary"))

    if risk["label"] != "benign":
        session_state.suspicious_turns += 1
    if marker_hits > 0:
        session_state.marker_turns += 1
    if canary_count > 0:
        session_state.unexpected_canary_turns += 1

    challenge = None
    response_text = None
    decision = "allow"

    should_deploy_honey = False
    if risk["label"] == "probing" and marker_hits > 0:
        should_deploy_honey = True
    elif canary_count > 0 and session_state.unexpected_canary_turns >= 2:
        should_deploy_honey = True
    elif repetition_count >= 2 and session_state.suspicious_turns >= 2:
        should_deploy_honey = True

    if session_state.issued_challenges:
        should_deploy_honey = False

    if should_deploy_honey:
        challenge = build_honey_challenge(
            session_state.session_id,
            query_text,
            risk,
            turn_index,
            retrieval_entry,
        )
        session_state.issued_challenges.append(challenge)
        decision = "deploy_honey"
        response_text = challenge.challenge_text
    elif risk["label"] != "benign":
        session_state.monitored_turns += 1
        decision = "monitor_only"
        response_text = _monitor_response(risk)
    else:
        response_text = "Normal response path allowed."

    return {
        "decision": decision,
        "risk_label": risk["label"],
        "risk_confidence": risk["confidence"],
        "reasons": risk["reasons"],
        "deployed_honey": challenge.__dict__ if challenge is not None else None,
        "honey_follow_up": honey_follow_up,
        "response_text": response_text,
        "policy_state": {
            "suspicious_turns": session_state.suspicious_turns,
            "marker_turns": session_state.marker_turns,
            "unexpected_canary_turns": session_state.unexpected_canary_turns,
            "monitored_turns": session_state.monitored_turns,
            "issued_challenges": len(session_state.issued_challenges),
        },
    }


def summarize_active_defense(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_turns = len(entries)
    honey_deployments = sum(1 for e in entries if e.get("decision") == "deploy_honey")
    monitored_turns = sum(1 for e in entries if e.get("decision") == "monitor_only")
    escalations = sum(1 for e in entries if e.get("decision") == "block_and_escalate")
    benign_escalations = sum(
        1
        for e in entries
        if e.get("decision") == "block_and_escalate" and str(e.get("session_profile")) == "benign"
    )

    session_outcomes: dict[str, dict[str, Any]] = {}
    for e in entries:
        session_id = str(e.get("session_id"))
        row = session_outcomes.setdefault(
            session_id,
            {
                "session_profile": e.get("session_profile"),
                "deployed_honey": False,
                "escalated": False,
            },
        )
        if e.get("decision") == "deploy_honey":
            row["deployed_honey"] = True
        if e.get("decision") == "block_and_escalate":
            row["escalated"] = True

    by_profile: dict[str, dict[str, int]] = {}
    for row in session_outcomes.values():
        profile = str(row.get("session_profile") or "unknown")
        stats = by_profile.setdefault(profile, {"sessions": 0, "honey_sessions": 0, "escalated_sessions": 0})
        stats["sessions"] += 1
        if row["deployed_honey"]:
            stats["honey_sessions"] += 1
        if row["escalated"]:
            stats["escalated_sessions"] += 1

    return {
        "total_turns": total_turns,
        "honey_deployments": honey_deployments,
        "monitored_turns": monitored_turns,
        "escalations": escalations,
        "benign_false_escalations": benign_escalations,
        "by_profile": by_profile,
    }


def format_honey_follow_up_query(template: str, challenge: HoneyChallenge) -> str:
    return template.format(
        honey_doc_id=challenge.honey_doc_id,
        honey_token=challenge.honey_token,
    )
