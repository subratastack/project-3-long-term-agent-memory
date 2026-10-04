"""Versioned fixture loading, auditable results, and evidence-only records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)

DATA = Path(__file__).parents[1] / "datasets" / "capabilities"


def implementation_fingerprint() -> str:
    """Fingerprint production and benchmark Python files, including uncommitted changes."""
    root = Path(__file__).parents[3]
    paths = sorted(
        [*root.glob("apps/memory_service/**/*.py"), *root.glob("apps/benchmark/**/*.py")]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def read_fixture(name: str) -> tuple[dict[str, Any], str]:
    path = DATA / f"{name}.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("version") != 1 or payload.get("track") != name:
        raise ValueError(f"unsupported fixture: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def answer_scores(answer: str, aliases: list[str]) -> dict[str, float]:
    """Diagnostic alias metrics; never a semantic or LLM judge."""
    predicted = normalize(answer)
    gold = [normalize(alias) for alias in aliases if normalize(alias)]
    exact = float(any(predicted == alias for alias in gold))
    contains = float(any(f" {alias} " in f" {predicted} " for alias in gold))
    pred_tokens = Counter(predicted.split())
    best_f1 = 0.0
    for alias in gold:
        gold_tokens = Counter(alias.split())
        overlap = sum((pred_tokens & gold_tokens).values())
        denominator = sum(pred_tokens.values()) + sum(gold_tokens.values())
        best_f1 = max(best_f1, 2 * overlap / denominator if denominator else 0.0)
    return {"exact_match": exact, "alias_containment": contains, "token_f1": best_f1}


def make_record(
    content: str,
    tenant: UUID,
    *,
    at: datetime,
    memory_type: MemoryType = MemoryType.EPISODIC,
    trust: TrustLevel = TrustLevel.SYSTEM,
    confidence: float = 0.9,
    valid_to: datetime | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    event: MemoryEvent | None = None,
    metadata: dict[str, Any] | None = None,
    subject_keys: list[str] | None = None,
) -> tuple[MemoryEvent, MemoryRecord]:
    event = event or MemoryEvent(
        tenant_id=tenant,
        source_type=SourceType.SYSTEM_EVENT,
        source_reference=f"capability:{uuid4()}",
        content=content,
        observed_at=at,
    )
    return event, MemoryRecord(
        tenant_id=tenant,
        content=content,
        confidence=confidence,
        provenance=[
            Provenance(
                event_id=event.event_id,
                source_type=event.source_type,
                source_reference=event.source_reference,
                observed_at=event.observed_at,
                trust_level=trust,
            )
        ],
        temporal_validity=TemporalValidity(valid_from=at, valid_to=valid_to),
        memory_type=memory_type,
        trust_level=trust,
        status=status,
        metadata=metadata or {},
        subject_keys=subject_keys or [],
        created_at=at,
        updated_at=at,
    )


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines)


def save_report(
    output: Path,
    track: str,
    payload: dict[str, Any],
    *,
    title: str,
    purpose: str,
    metrics: str,
    results: str,
    interpretation: str,
    limitations: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "track": track,
        "created_at": datetime.now(UTC).isoformat(),
        "implementation_sha256": implementation_fingerprint(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        **payload,
    }
    (output / f"{track}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    guide_link = os.path.relpath(
        Path(__file__).parents[3] / "docs/capability-benchmarks.md", start=output.resolve()
    )
    text = (
        f"# {title}\n\n{purpose}\n\n"
        f"These are measured results from {payload['created_at']}. "
        "The JSON companion contains settings, fixture hashes, and individual outcomes.\n\n"
        f"## What the metrics mean\n\n{metrics}\n\n## Measured results\n\n{results}\n\n"
        f"## What we learned\n\n{interpretation}\n\n"
        f"## Scope and next experiment\n\n{limitations}\n\n"
        f"[Full results]({track}.json) · [Benchmark guide]({guide_link})\n"
    )
    (output / f"{track}.md").write_text(text)
