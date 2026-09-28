"""Tests for `OllamaCandidateExtractor` against a mocked HTTP transport.

No real Ollama server is involved: `httpx.MockTransport` intercepts the
request and returns a canned response, so these tests exercise request
shaping and response validation only.
"""

import json
import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx

from apps.memory_service.domain.enums import SourceType
from apps.memory_service.domain.models import MemoryEvent
from apps.memory_service.ingestion.candidate_extractor import OllamaCandidateExtractor

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _make_event(**overrides: Any) -> MemoryEvent:
    defaults: dict[str, Any] = {
        "tenant_id": TENANT_ID,
        "source_type": SourceType.TOOL_OUTPUT,
        "source_reference": "tool-1",
        "content": "connection pool exhausted",
        "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return MemoryEvent(**defaults)


def _client_returning(ollama_response_text: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": ollama_response_text})

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestOllamaCandidateExtractor(unittest.TestCase):

    def test_valid_json_response_is_parsed_into_raw_candidates(self) -> None:
        event = _make_event()
        payload = [
            {
                "content": "The connection pool was exhausted.",
                "source_event_ids": [str(event.event_id)],
                "memory_type_hint": "episodic",
                "confidence": 0.7,
                "subject_keys": ["connection pool"],
            }
        ]
        extractor = OllamaCandidateExtractor(
            model="llama3", client=_client_returning(json.dumps(payload))
        )

        candidates = extractor.extract([event])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "The connection pool was exhausted.")
        self.assertEqual(candidates[0].source_event_ids, [event.event_id])

    def test_object_wrapped_candidates_response_is_parsed(self) -> None:
        # Ollama's `format: "json"` constrains the grammar toward a
        # top-level object, so real responses arrive as
        # {"candidates": [...]}  rather than a bare array.
        event = _make_event()
        payload = {
            "candidates": [
                {
                    "content": "The connection pool was exhausted.",
                    "source_event_ids": [str(event.event_id)],
                }
            ]
        }
        extractor = OllamaCandidateExtractor(
            model="llama3", client=_client_returning(json.dumps(payload))
        )

        candidates = extractor.extract([event])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "The connection pool was exhausted.")

    def test_object_response_missing_candidates_key_yields_none(self) -> None:
        event = _make_event()
        extractor = OllamaCandidateExtractor(
            model="llama3", client=_client_returning(json.dumps({"unexpected": "shape"}))
        )

        self.assertEqual(extractor.extract([event]), [])

    def test_non_json_response_yields_no_candidates(self) -> None:
        event = _make_event()
        extractor = OllamaCandidateExtractor(
            model="llama3", client=_client_returning("not json at all")
        )

        self.assertEqual(extractor.extract([event]), [])

    def test_response_element_failing_validation_is_dropped_not_raised(self) -> None:
        event = _make_event()
        payload = [
            {"content": "valid one", "source_event_ids": [str(event.event_id)]},
            {"source_event_ids": [str(event.event_id)]},  # missing required "content"
        ]
        extractor = OllamaCandidateExtractor(
            model="llama3", client=_client_returning(json.dumps(payload))
        )

        candidates = extractor.extract([event])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "valid one")

    def test_no_events_short_circuits_without_a_request(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"response": "[]"})

        extractor = OllamaCandidateExtractor(
            model="llama3", client=httpx.Client(transport=httpx.MockTransport(handler))
        )

        self.assertEqual(extractor.extract([]), [])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
