from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class CanaryDesign:
    keyword_token: str
    semantic_token: str


def generate_keyword_canaries(design: CanaryDesign, count: int = 5) -> list[str]:
    """
    Generate keyword-dense canary documents.

    Properties:
    - Contain a rare, highly distinctive token many times (for boosting/poisoning detection).
    - The surrounding content is about obscure, synthetic topics unlikely to match benign queries.
    """
    docs: list[str] = []
    base_paragraph = (
        "This internal synthetic document describes a fictitious protocol used only for security testing. "
        "The content is intentionally obscure and unrelated to any real product, customer, or workflow. "
        "It repeatedly references the internal marker {token} to make retrieval highly sensitive to its presence. "
        "No benign query should naturally mention this marker or the fictional protocol it describes."
    )
    for i in range(count):
        payload = base_paragraph.format(token=design.keyword_token)
        # Make it keyword-dense by repeating the token.
        tail = " ".join([design.keyword_token] * 20)
        docs.append(payload + "\n\n" + tail)
    return docs


def generate_semantic_canaries(design: CanaryDesign, count: int = 5) -> list[str]:
    """
    Generate semantically targeted canary documents.

    Properties:
    - Look like normal knowledge base articles (structure, tone).
    - Use vocabulary aligned with semantic probing and membership-inference style attacks.
    - Avoid broad operational/helpdesk phrasing that could surface for benign queries.
    - Embed a rare token in a way that is semantically neutral but machine-detectable.
    """
    docs: list[str] = []
    templates = [
        (
            "Internal evaluation note: some synthetic records are retained only for security testing and "
            "marker-registry validation. These records describe dormant internal testing protocol identifiers, "
            "isolated validation artifacts, and non-production archive entries that should never appear in normal "
            "customer workflows. One unobtrusive marker phrase, {token}, is preserved so targeted retrieval or "
            "membership-inference attempts can be detected."
        ),
        (
            "Restricted audit entry: this document summarizes how the least frequently accessed internal material "
            "is reviewed during search-security assessments. The content references synthetic documents, dormant "
            "registry items, and isolated test artifacts that exist only to measure unusual retrieval behavior. "
            "As part of that process, the archived marker {token} is retained in the record."
        ),
        (
            "Security validation memo: analysts sometimes seed clearly synthetic corpus entries that look ordinary "
            "but are used only for red-team probing studies. These entries mention hidden marker registries, "
            "internal testing protocol identifiers, and low-visibility documents that should not surface during "
            "benign use. The marker {token} appears here as a controlled evaluation artifact."
        ),
        (
            "Search integrity checklist: when reviewing whether anyone is enumerating synthetic or security-only "
            "documents, auditors look for requests about internal testing protocol identifiers, hidden marker lists, "
            "or the least frequently accessed internal records. This otherwise routine note includes the evaluation "
            "marker {token} for controlled tracing."
        ),
        (
            "Private archive summary: a subset of security-testing documents is intentionally rare, synthetic, and "
            "isolated from ordinary support or operations workflows. Their purpose is to reveal probing for hidden "
            "identifiers, synthetic content, or dormant registry entries. This archive summary keeps the token "
            "{token} as part of the controlled record."
        ),
    ]
    for i in range(count):
        docs.append(templates[i % len(templates)].format(token=design.semantic_token))
    return docs


def write_canary_corpus(
    output_dir: Path,
    design: CanaryDesign,
    keyword_count: int = 5,
    semantic_count: int = 5,
) -> None:
    """
    Materialize canary documents as plain text files.

    Files are named:
    - keyword_canary_XX.txt
    - semantic_canary_XX.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_docs = generate_keyword_canaries(design, keyword_count)
    for idx, text in enumerate(keyword_docs):
        path = output_dir / f"keyword_canary_{idx:02d}.txt"
        path.write_text(text, encoding="utf-8")

    semantic_docs = generate_semantic_canaries(design, semantic_count)
    for idx, text in enumerate(semantic_docs):
        path = output_dir / f"semantic_canary_{idx:02d}.txt"
        path.write_text(text, encoding="utf-8")
