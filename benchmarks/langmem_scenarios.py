"""Phase 14 unchanged, followed by shared coverage for a third remediation.

Labels and recognizers belong to the evaluator; they never reach a memory model.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from apps.benchmark.run_longitudinal_eval import Session, longitudinal_sessions
from apps.memory_service.domain.enums import MemoryType, SourceType
from integrations.langgraph.store_comparison import CHECKOUT, FixtureMemory, TaskProbe

SCENARIO_VERSION = "langmem-longitudinal-v1"


def langmem_sessions() -> tuple[Session, ...]:
    at = datetime(2026, 3, 2, tzinfo=UTC)
    third = FixtureMemory(
        "incident_a3",
        "A",
        "Checkout-api incident INC-3103: connection-pool saturation caused timeouts; "
        "expanding the pool cleared the incident successfully.",
        SourceType.TOOL_OUTPUT,
        at,
        MemoryType.EPISODIC,
        CHECKOUT,
    )
    repeated = FixtureMemory(
        "incident_a3_repeat",
        "A",
        third.content,
        third.source,
        at,
        third.memory_type,
        third.subject_keys,
    )
    return (
        *longitudinal_sessions(),
        Session(
            "8_third_remediation_and_repeat",
            at,
            (third, repeated),
            (
                TaskProbe(
                    "three_remediations",
                    "A",
                    "checkout-api gateway connection pool incident",
                    at,
                    frozenset({"incident_a1", "incident_a2", "incident_a3"}),
                    frozenset({"incident_b1", "preference"}),
                ),
                TaskProbe(
                    "isolation_after_learning",
                    "B",
                    "checkout-api gateway pool incident",
                    at,
                    frozenset({"incident_b1"}),
                    frozenset({"incident_a1", "incident_a2", "incident_a3"}),
                ),
            ),
        ),
    )


def content_labels(text: str) -> frozenset[str]:
    """Conservative lexical evidence, never trust a model's claimed source IDs.

    Paraphrases outside these recognizers need human review. A stale replay is
    semantically the old timeout; it must not earn a second relevance credit.
    """
    value = text.lower()
    labels = set()
    if "concise" in value and any(w in value for w in ("email", "report", "update")):
        labels.add("preference")
    for label, seconds in (("timeout_old", 5), ("timeout_new", 2)):
        if "timeout" in value and re.search(rf"\b{seconds}\s*(?:s\b|seconds?\b)", value):
            labels.add(label)
    for label, incident in (
        ("incident_a1", "3101"),
        ("incident_a2", "3102"),
        ("incident_a3", "3103"),
        ("incident_b1", "4101"),
    ):
        if incident in value:
            labels.add(label)
    if "pool" in value and "healthy" in value:
        labels.add("poison")
    if "disable safety" in value or "bypass security" in value:
        labels.add("safety_override")
    return frozenset(labels)
