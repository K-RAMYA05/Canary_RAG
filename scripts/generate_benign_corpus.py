from __future__ import annotations

import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRODUCT_NAMES = [
    "Atlas",
    "Nimbus",
    "Helios",
    "Orion",
    "Nova",
]

REGIONS = [
    "us-east",
    "us-west",
    "eu-central",
    "ap-south",
]

TEAMS = [
    "Core Platform",
    "Data Engineering",
    "Site Reliability",
    "Support Operations",
    "Security Engineering",
]

OWNERS = [
    "alice@example.internal",
    "bob@example.internal",
    "carol@example.internal",
    "dave@example.internal",
    "eve@example.internal",
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

INCIDENT_STATUSES = ["detected", "mitigated", "resolved", "monitoring"]

IMPACTED_AREAS = [
    "public API",
    "authentication service",
    "reporting pipeline",
    "dashboard rendering",
    "background jobs",
]

TAGS_BY_TYPE = {
    "faq": ["support", "how-to", "user-facing"],
    "guide": ["onboarding", "runbook", "internal-doc"],
    "design": ["architecture", "technical-design", "engineering"],
    "incident": ["incident", "postmortem", "reliability"],
    "changelog": ["release", "versioning", "deployment"],
    "policy": ["compliance", "security", "governance"],
}


FAQ_TEMPLATES = [
    "Q: How do I reset my account password?\n"
    "A: Navigate to the account settings page, choose 'Security', and click on "
    "'Reset Password'. You will receive a one-time link by email. The link "
    "expires after 30 minutes, so complete the reset promptly.\n",
    "Q: What should I do if I see a 500 error?\n"
    "A: First, refresh the page to ensure it was not a transient issue. If the "
    "problem persists, capture the time, URL, and any steps to reproduce. Open "
    "an incident ticket with this information so the on-call engineer can "
    "investigate logs and metrics.\n",
    "Q: How can I request access to a new workspace?\n"
    "A: Submit an access request through the internal portal, including the "
    "business justification and project name. Your manager will receive an "
    "approval task, and access will be granted automatically once it is "
    "approved.\n",
    "Q: Where can I find system status information?\n"
    "A: Check the internal status page, which displays the health of major "
    "services, recent incidents, and any ongoing maintenance windows.\n",
]


GUIDE_TEMPLATES = [
    "This guide walks through deploying the application to the staging "
    "environment. Start by creating a release branch, running the full "
    "test suite, and confirming that any database migrations have been "
    "reviewed. Once tests pass, use the deployment tool to promote the "
    "artifact and monitor health checks.\n",
    "To onboard a new engineer, make sure they have access to version control, "
    "continuous integration dashboards, and the documentation portal. Provide "
    "an overview of the architecture and a simple starter task that exercises "
    "the main workflows without touching production systems.\n",
    "Use feature flags to safely roll out changes. Start with a small "
    "percentage of internal users, monitor metrics and logs, and gradually "
    "increase exposure as confidence grows. Document rollback procedures in "
    "the deployment checklist.\n",
    "When debugging a production issue, begin by checking alerts, logs, and "
    "dashboards. Reproduce the issue in a lower environment when possible, "
    "and communicate status updates in the incident channel.\n",
]


DESIGN_TEMPLATES = [
    "The system is organized into several services connected through a message "
    "bus. Each service owns its own data storage and exposes a clear API. "
    "Requests from the user interface flow through an edge gateway, which "
    "handles authentication, rate limiting, and routing to backend services.\n",
    "Caching is introduced at multiple layers to reduce latency. The client "
    "caches stable configuration data, while the edge gateway caches frequently "
    "requested resources with short time-to-live values. Backend services "
    "maintain their own caches for expensive computations.\n",
    "Each service publishes metrics for latency, error rate, and throughput. "
    "Dashboards aggregate these metrics to provide a high-level view, while "
    "detailed panels help engineers drill into specific components.\n",
    "The data model is designed to support gradual evolution. Schemas use "
    "additive changes where possible, and deprecations are communicated before "
    "fields are removed. Migrations are tested in staging before production.\n",
]


INCIDENT_TEMPLATES = [
    "Incident report summary: An elevated error rate was observed in the "
    "primary API during peak traffic. The immediate impact was that some "
    "users experienced timeouts while loading dashboards. The response "
    "included rolling back a configuration change and adding temporary logging "
    "to narrow down the root cause.\n",
    "Post-incident review: The outage was traced to a misconfigured database "
    "connection pool. The pool size was reduced during routine maintenance, "
    "which caused contention under normal peak load. The fix involved "
    "restoring appropriate pool parameters and adding a pre-deployment check.\n",
    "Timeline: The first alert fired on the metrics dashboard, followed by "
    "customer reports in the support channel. Engineers confirmed the issue, "
    "identified the suspect deployment, and coordinated a rollback while "
    "monitoring error rates and latency.\n",
    "Follow-up actions: Add automated checks for configuration drift, improve "
    "alert descriptions to reduce ambiguity, and document decision points from "
    "the incident to refine the response playbook.\n",
]


CHANGELOG_TEMPLATES = [
    "Release notes: This version includes improvements to the search interface, "
    "more precise filters, and better handling of time ranges. Several minor "
    "bugs were fixed, including alignment issues in the navigation bar and "
    "inconsistent error messages reported by the API.\n",
    "Performance updates: Batch processing jobs now run in parallel, and "
    "redundant queries have been removed. The user interface displays progress "
    "while long reports are generated, and streaming responses are used for "
    "expensive computations.\n",
    "Developer experience: New command-line tools assist with local testing, "
    "database migrations, and feature flag management. Documentation for the "
    "build pipeline has been expanded with examples and troubleshooting tips.\n",
    "Deprecations: Several legacy endpoints have been marked as deprecated. "
    "Clients are encouraged to migrate to the newer, versioned APIs before the "
    "next major release.\n",
]


POLICY_TEMPLATES = [
    "The acceptable use policy requires that all employees protect account "
    "credentials and avoid sharing confidential information in public forums. "
    "Access to systems must be tied to individual accounts, and shared "
    "passwords are not permitted.\n",
    "The data retention policy describes how long different categories of "
    "information are stored. Operational logs are kept for troubleshooting and "
    "audit purposes, while personally identifiable information is retained only "
    "for as long as necessary to provide the service.\n",
    "The incident response policy outlines how to declare an incident, who to "
    "notify, and what roles are responsible for coordination, communications, "
    "and technical remediation. Regular drills are recommended to keep the "
    "process familiar.\n",
    "The access review policy describes how managers periodically verify that "
    "team members have appropriate permissions. Access that is no longer "
    "required should be revoked promptly to reduce risk.\n",
]


def synth_paragraphs(paragraphs: list[str], min_paragraphs: int, max_paragraphs: int) -> str:
    """
    Build a document body from unique paragraphs (no repeats within a single doc).
    """
    if not paragraphs:
        return ""
    max_paragraphs = min(max_paragraphs, len(paragraphs))
    min_paragraphs = min(min_paragraphs, max_paragraphs)
    count = random.randint(min_paragraphs, max_paragraphs)
    chosen = random.sample(paragraphs, k=count)
    return "\n".join(chosen)


def make_header(doc_type: str, doc_index: int) -> str:
    product = random.choice(PRODUCT_NAMES)
    region = random.choice(REGIONS)
    team = random.choice(TEAMS)
    doc_id = f"{doc_type.upper()}-{doc_index:05d}"
    owner = random.choice(OWNERS)

    if doc_type == "faq":
        prefix = "Support FAQ"
    elif doc_type == "guide":
        prefix = "How-to Guide"
    elif doc_type == "design":
        prefix = "Design Note"
    elif doc_type == "incident":
        prefix = "Incident Document"
    elif doc_type == "changelog":
        prefix = "Release Changelog"
    elif doc_type == "policy":
        prefix = "Internal Policy"
    else:
        prefix = "Documentation"

    created_at = datetime(2023, 1, 1) + timedelta(days=doc_index)
    created_str = created_at.strftime("%Y-%m-%d")
    last_updated = created_at + timedelta(days=random.randint(0, 45))
    last_updated_str = last_updated.strftime("%Y-%m-%d")

    base_tags = TAGS_BY_TYPE.get(doc_type, ["documentation"])
    tag_sample_size = min(len(base_tags), 3)
    tags = ", ".join(random.sample(base_tags, k=tag_sample_size))

    header_lines = [
        f"{prefix} for {product} in {region} maintained by {team}.",
        f"Internal reference: {doc_id}.",
        f"Owner: {owner}.",
        f"Created on: {created_str}.",
        f"Last updated: {last_updated_str}.",
        f"Tags: {tags}.",
    ]

    # Extra structured metadata for log-like documents.
    if doc_type in ("incident", "changelog"):
        severity = random.choice(SEVERITY_LEVELS)
        area = random.choice(IMPACTED_AREAS)
        duration_minutes = random.randint(5, 240)
        status = random.choice(INCIDENT_STATUSES)
        header_lines.extend(
            [
                f"Severity: {severity}.",
                f"Impacted area: {area}.",
                f"Duration (minutes): {duration_minutes}.",
                f"Status: {status}.",
            ]
        )

    header = "\n".join(header_lines) + "\n"
    return header


def generate_doc(doc_type: str, doc_index: int) -> str:
    """
    Generate a single benign document of a given type.

    Each document gets a unique synthetic header (product, region, team, id)
    plus a body assembled from topic-specific paragraphs. This makes documents
    differ across files even when they share some underlying paragraph templates.
    """
    if doc_type == "faq":
        body = synth_paragraphs(FAQ_TEMPLATES, 3, 5)
    elif doc_type == "guide":
        body = synth_paragraphs(GUIDE_TEMPLATES, 3, 5)
    elif doc_type == "design":
        body = synth_paragraphs(DESIGN_TEMPLATES, 3, 5)
    elif doc_type == "incident":
        body = synth_paragraphs(INCIDENT_TEMPLATES, 3, 4)
    elif doc_type == "changelog":
        body = synth_paragraphs(CHANGELOG_TEMPLATES, 3, 4)
    elif doc_type == "policy":
        body = synth_paragraphs(POLICY_TEMPLATES, 3, 5)
    else:
        body = synth_paragraphs(GUIDE_TEMPLATES, 3, 5)

    header = make_header(doc_type, doc_index)
    return header + "\n" + body


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic benign text corpus.")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=200,
        help="Total number of documents to generate.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="general",
        choices=["general", "support", "engineering", "policy", "balanced"],
        help="Skew the corpus toward a particular content profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "data" / "corpus"),
        help="Directory where benign documents will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    corpus_dir = Path(args.output_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    profile_to_types = {
        # Strongly bias general corpora toward logs (incident/changelog)
        "general": [
            "faq",
            "guide",
            "design",
            "incident",
            "incident",
            "incident",
            "changelog",
            "changelog",
            "changelog",
            "policy",
        ],
        "support": ["faq", "guide", "incident"],
        "engineering": ["design", "design", "guide", "changelog", "changelog"],
        "policy": ["policy"],
        "balanced": ["incident", "changelog", "faq", "guide", "design", "policy"],
    }
    doc_types = profile_to_types[args.profile]

    for i in range(args.num_docs):
        if args.profile == "balanced":
            doc_type = doc_types[i % len(doc_types)]
        else:
            doc_type = random.choice(doc_types)
        text = generate_doc(doc_type, i)
        path = corpus_dir / f"{doc_type}_{i:04d}.txt"
        path.write_text(text, encoding="utf-8")

    print(f"Generated {args.num_docs} benign documents under {corpus_dir} (profile={args.profile})")


if __name__ == "__main__":
    main()
