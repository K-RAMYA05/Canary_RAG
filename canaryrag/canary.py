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
    Generate semantically broad canary documents.

    Properties:
    - Look like normal knowledge base articles (structure, tone).
    - Discuss generic topics (e.g., project management, documentation hygiene).
    - Embed a rare token in a way that is semantically neutral but machine-detectable.
    """
    docs: list[str] = []
    base_paragraph = (
        "This document discusses general best practices for knowledge management, including version control, "
        "documentation hygiene, and principles for designing internal search systems. It is intentionally generic "
        "and should be semantically similar to many real documents in a technical organization. As part of an "
        "internal evaluation, it embeds an unobtrusive marker phrase, {token}, which does not affect the meaning "
        "of the text but can be used to detect targeted retrieval or membership inference attacks."
    )
    for i in range(count):
        docs.append(base_paragraph.format(token=design.semantic_token))
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

