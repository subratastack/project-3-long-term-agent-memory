"""Structured candidate extraction: the first stage of the ingestion pipeline.

An extractor turns raw `MemoryEvent`s into `RawCandidate` proposals -- content
that *might* become a memory, but that is not yet classified, normalized, or
policy-evaluated. Nothing here is trusted: a `RawCandidate` is exactly as
untrusted as the event(s) it was built from, whether it came from a
hand-written rule (`FakeCandidateExtractor`, used in tests) or an LLM
(`OllamaCandidateExtractor`). See ARCHITECTURE.md's "Write path" diagram --
extraction feeds `classifier.classify_memory_type` and then
`normalizer.normalize_candidate`, and only the result of that chain is ever
handed to `apps.memory_service.ingestion.write_policy`.

The LLM proposes; it never decides. `evaluate_write_policy` remains the only
component allowed to approve a write, regardless of which extractor produced
the candidate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.memory_service.domain.enums import MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryEvent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from apps.memory_service.ingestion.normalizer import RejectedExtraction


class RawCandidate(BaseModel):
    """An extractor's untrusted, pre-classification proposal for a memory.

    This is deliberately looser than `MemoryCandidate`: it has no
    `memory_type` yet (that is `classifier.classify_memory_type`'s job), its
    `source_event_ids` may be empty (an extractor that failed to ground its
    own output), and its `proposed_trust_level` is only a hint -- the trust
    level that actually ends up on a memory is always re-derived from the
    real source events in `normalizer.normalize_candidate` and
    `ingestion.provenance.verify_provenance`, never taken from here.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(
        min_length=1,
        description="Raw, not-yet-normalized text of the proposed memory.",
    )
    source_event_ids: list[UUID] = Field(
        default_factory=list,
        description=(
            "Events this proposal claims to be grounded in. Empty means the "
            "extractor produced content it cannot tie back to any evidence."
        ),
    )
    memory_type_hint: MemoryType | None = Field(
        default=None,
        description="The extractor's own guess at memory type; advisory only.",
    )
    proposed_trust_level: TrustLevel | None = Field(
        default=None,
        description="The extractor's own guess at trust; advisory only.",
    )
    subject_keys: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateExtractor(Protocol):
    """The one method every extractor -- fake, rule-based, or LLM -- implements."""

    def extract(self, events: Sequence[MemoryEvent]) -> list[RawCandidate]:
        """Propose zero or more `RawCandidate`s grounded in `events`."""
        ...


class FakeCandidateExtractor:
    """A test double driven by a plain rule function, not an LLM.

    `rule` receives the exact `events` passed to `extract` and returns the
    `RawCandidate`s to propose for them. Tests use this to exercise
    classification, normalization, and write policy end to end without ever
    calling Ollama.

    Example:
        Input:
            rule = lambda events: [RawCandidate(content="timeout=2s",
                                                 source_event_ids=[events[0].event_id])]
            extractor = FakeCandidateExtractor(rule)
        Output:
            extractor.extract(events) == [RawCandidate(content="timeout=2s", ...)]
    """

    def __init__(self, rule: Callable[[Sequence[MemoryEvent]], list[RawCandidate]]) -> None:
        self._rule = rule

    def extract(self, events: Sequence[MemoryEvent]) -> list[RawCandidate]:
        return self._rule(events)


def build_memory_candidates(
    extractor: CandidateExtractor,
    events: Sequence[MemoryEvent],
) -> list[Any | RejectedExtraction]:
    """Run the full extract -> classify -> normalize chain for `events`.

    How it works:
        1. Ask `extractor` for its raw proposals.
        2. For each proposal, run the deterministic `classify_memory_type`
           (never the extractor's own `memory_type_hint`) to get a
           `MemoryType`.
        3. Run `normalize_candidate`, which either returns a fully-formed
           `MemoryCandidate` ready for `write_policy.evaluate_write_policy`,
           or a `RejectedExtraction` if the proposal cannot be grounded in
           real evidence (e.g. no `source_event_ids`) -- a rejection that
           never touches write policy at all.

        The imports below are local to this function, not module-level: both
        `classifier` and `normalizer` need `RawCandidate` from this module,
        so a module-level import here would create an import cycle.

    Example:
        Input:
            extractor = FakeCandidateExtractor(lambda evs: [RawCandidate(
                content="timeout=2s", source_event_ids=[evs[0].event_id])])
            events = [<a CONFIGURATION MemoryEvent>]
        Output:
            [MemoryCandidate(content="timeout=2s", memory_type=MemoryType.SEMANTIC, ...)]
    """
    from apps.memory_service.ingestion.classifier import classify_memory_type
    from apps.memory_service.ingestion.normalizer import normalize_candidate

    outcomes = []
    for raw in extractor.extract(events):
        memory_type = classify_memory_type(raw, events)
        outcomes.append(normalize_candidate(raw, memory_type, events))
    return outcomes


class OllamaCandidateExtractor:
    """A `CandidateExtractor` backed by a local Ollama model.

    The model is asked to return a JSON array of candidate objects; each one
    is validated against `RawCandidate` before it is trusted to exist at all.
    A response that is not valid JSON, or an element that fails `RawCandidate`
    validation, is dropped rather than raised -- an extractor's job is only to
    *propose*, so a malformed LLM response degrades to "no candidates" rather
    than crashing the pipeline. Nothing here bypasses `classify_memory_type`,
    `normalize_candidate`, or `evaluate_write_policy`: this class only ever
    produces the same untrusted `RawCandidate` a fake extractor would.
    """

    # Ollama's `format: "json"` constrains the grammar toward a top-level JSON
    # *object*, not a bare array -- under that constraint, weaker instruct
    # models reliably degenerate to `{}` if asked for an array directly. A
    # `{"candidates": [...]}` wrapper, plus one worked example, is what
    # actually gets small models (verified against qwen2.5:7b-instruct) to
    # populate the array instead of defaulting to empty.
    _SYSTEM_PROMPT = (
        "You extract candidate long-term memories from agent event logs. "
        'Respond with ONLY a JSON object shaped like {"candidates": '
        '[{"content": str, "source_event_ids": [str], "memory_type_hint": '
        '"episodic"|"semantic"|"procedural"|null, "confidence": float, '
        '"subject_keys": [str]}]}. Use only event ids you were given, copied '
        "exactly. Extract at least one candidate whenever an event describes "
        "a fact, outcome, or configuration worth remembering. Example: given "
        "event_id=X source_type=configuration content=timeout=5s, return "
        '{"candidates": [{"content": "The configured timeout is 5 seconds.", '
        '"source_event_ids": ["X"], "memory_type_hint": "semantic", '
        '"confidence": 0.9, "subject_keys": ["timeout"]}]}. If nothing is '
        'worth remembering, return {"candidates": []}.'
    )

    def __init__(
        self,
        *,
        model: str,
        client: Any,
        base_url: str = "http://localhost:11434",
    ) -> None:
        """`client` is an `httpx.Client`-shaped object (injected for testability)."""
        self._model = model
        self._client = client
        self._base_url = base_url.rstrip("/")

    def extract(self, events: Sequence[MemoryEvent]) -> list[RawCandidate]:
        if not events:
            return []
        prompt = self._build_prompt(events)
        logger.debug("ollama request model=%s prompt=%r", self._model, prompt)
        response = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "system": self._SYSTEM_PROMPT,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ollama response:\n%s", _pretty_json(raw_text))
        candidates = self._parse_candidates(raw_text)
        logger.info(
            "ollama extracted %d candidate(s) from %d event(s)", len(candidates), len(events)
        )
        return candidates

    @staticmethod
    def _build_prompt(events: Sequence[MemoryEvent]) -> str:
        lines = [
            f"- event_id={event.event_id} source_type={event.source_type} "
            f"content={event.content!r}"
            for event in events
        ]
        return "Events:\n" + "\n".join(lines)

    @staticmethod
    def _parse_candidates(raw_text: str) -> list[RawCandidate]:
        try:
            payload = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(payload, dict):
            payload = payload.get("candidates", [])
        if not isinstance(payload, list):
            return []

        candidates: list[RawCandidate] = []
        for item in payload:
            try:
                candidates.append(RawCandidate.model_validate(item))
            except ValidationError:
                continue
        return candidates


def _pretty_json(raw_text: str) -> str:
    """Indent `raw_text` for logging if it's valid JSON, else return it as-is.

    Ollama's response is not guaranteed to be valid JSON (a small model can
    still emit malformed output despite `format: "json"`), so this is a
    display nicety only -- `OllamaCandidateExtractor._parse_candidates` does
    its own, independent parsing and never relies on this function.
    """
    try:
        return json.dumps(json.loads(raw_text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return raw_text
