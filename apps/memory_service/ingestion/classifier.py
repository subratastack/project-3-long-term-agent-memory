"""Deterministic classification of a `RawCandidate` into a `MemoryType`.

Like `write_policy`, this is intentionally rule-based rather than model-based:
an extractor (fake, rule-based, or LLM-backed) may attach its own
`memory_type_hint`, but that hint is advisory only and this module never
takes it at face value for anything but the procedural check below, where a
hint can only ever push a decision *toward* the stricter type, never away
from it. See ARCHITECTURE.md's "Memory classification and promotion" table.
"""

import re
from collections.abc import Sequence

from apps.memory_service.domain.enums import MemoryType, SourceType
from apps.memory_service.domain.models import MemoryEvent
from apps.memory_service.ingestion.candidate_extractor import RawCandidate

# A bare `key = value` shape (e.g. "timeout=2s", "retries = 3") is the
# fingerprint of a configuration fact worth distilling into semantic memory.
_ASSIGNMENT_PATTERN = re.compile(r"[A-Za-z_][\w.]*\s*=\s*\S+")

# Phrases that mark content as describing a repeatable *procedure* rather
# than a one-off observation -- deliberately simple substring matches, kept
# in one auditable list rather than inferred by a model.
_PROCEDURE_PHRASES: tuple[str, ...] = (
    "diagnostic sequence",
    "runbook",
    "step 1",
    "first, ",
    "then, ",
    "always ",
    "procedure",
)


def classify_memory_type(raw: RawCandidate, events: Sequence[MemoryEvent]) -> MemoryType:
    """Deterministically classify `raw` as EPISODIC, SEMANTIC, or PROCEDURAL.

    How it works:
        1. Resolve `raw.source_event_ids` against `events` (unresolved ids are
           silently skipped here; `normalizer.normalize_candidate` is what
           turns an unresolvable id into a rejection).
        2. SEMANTIC wins first: any source event from a `CONFIGURATION`
           source, or content that looks like a `key=value` assignment,
           describes a durable fact rather than something that happened.
        3. PROCEDURAL is next: an explicit `memory_type_hint`, more than one
           distinct source event (a multi-step sequence), or a procedural
           keyword in the content, all suggest a repeatable routine rather
           than a single observation.
        4. Otherwise, default to EPISODIC -- a concrete thing that happened,
           the safest default for content that doesn't clearly fit the other
           two categories.

    Example:
        Input:
            raw.content = "The configured request timeout is 2s."
            events = [MemoryEvent(source_type=SourceType.CONFIGURATION, ...)]
        Output:
            MemoryType.SEMANTIC
    """
    events_by_id = {event.event_id: event for event in events}
    source_events = [
        events_by_id[event_id] for event_id in raw.source_event_ids if event_id in events_by_id
    ]

    if _looks_like_configuration(raw, source_events):
        return MemoryType.SEMANTIC

    if _looks_like_procedure(raw, source_events):
        return MemoryType.PROCEDURAL

    return MemoryType.EPISODIC


def _looks_like_configuration(raw: RawCandidate, source_events: list[MemoryEvent]) -> bool:
    if any(event.source_type is SourceType.CONFIGURATION for event in source_events):
        return True
    return bool(_ASSIGNMENT_PATTERN.search(raw.content))


def _looks_like_procedure(raw: RawCandidate, source_events: list[MemoryEvent]) -> bool:
    if raw.memory_type_hint is MemoryType.PROCEDURAL:
        return True
    if len({event.event_id for event in source_events}) > 1:
        return True
    haystack = raw.content.lower()
    return any(phrase in haystack for phrase in _PROCEDURE_PHRASES)
