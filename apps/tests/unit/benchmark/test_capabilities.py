"""Scoring failures must stay visible instead of producing misleading benchmark scores."""

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from apps.benchmark.capabilities.common import answer_scores, read_fixture
from apps.benchmark.capabilities.model_tracks import match_facts, parse_answer, run_reasoning
from apps.benchmark.capabilities.ollama import OllamaClient
from apps.benchmark.capabilities.system_tracks import remap_uuids


def test_answer_matching_respects_word_boundaries_and_duplicate_tokens() -> None:
    assert answer_scores("12 seconds", ["2 seconds"])["alias_containment"] == 0
    assert answer_scores("The timeout is 2 seconds.", ["2 seconds"])["alias_containment"] == 1
    assert answer_scores("pool pool", ["pool size"])["token_f1"] == 0.5
    assert answer_scores("", ["yes"])["exact_match"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        "not JSON",
        '{"answer": 240}',
        '{"answer":"240","citations":[],"abstain":"false"}',
        '{"answer":"240","citations":[12],"abstain":false}',
    ],
)
def test_malformed_answer_is_not_scored_as_a_valid_completion(raw: str) -> None:
    assert parse_answer(raw) is None


def test_fact_assignment_is_maximum_matching_and_cannot_reward_duplicates() -> None:
    facts = [{"groups": [["timeout"]]}, {"groups": [["pool"]]}]
    assigned = match_facts(["timeout and pool", "timeout"], facts)
    assert assigned == {0: 1, 1: 0}
    assert len(match_facts(["timeout", "timeout"], facts)) == 1


def test_numeric_final_answer_is_normalized_but_boolean_answer_is_rejected() -> None:
    parsed = parse_answer('{"answer":240,"citations":["M1"],"abstain":false}')
    assert parsed is not None and parsed["answer"] == "240"
    assert parse_answer('{"answer":true,"citations":[],"abstain":false}') is None


def test_uuid_namespace_keeps_evidence_edges_and_foreign_tenants_distinct() -> None:
    mine, theirs, event = str(uuid4()), str(uuid4()), str(uuid4())
    source = {"tenant": mine, "foreign": theirs, "event": {"id": event}, "cites": [event]}
    remapped = remap_uuids(source, uuid4())
    assert remapped["tenant"] != mine
    assert remapped["tenant"] != remapped["foreign"]
    assert remapped["event"]["id"] == remapped["cites"][0]
    assert source["tenant"] == mine


def test_every_component_has_an_independent_versioned_fixture() -> None:
    tracks = (
        "answer_quality",
        "reasoning",
        "memory_extraction",
        "write_policy",
        "conflict_resolution",
        "forgetting",
        "prompt_packing",
    )
    hashes = [read_fixture(track)[1] for track in tracks]
    assert len(set(hashes)) == 7


def test_reasoning_errors_remain_in_denominator_and_thinking_is_not_saved(tmp_path: Path) -> None:
    data, _ = read_fixture("reasoning")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "local", "digest": "test", "details": {}}]}
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["thinking"]})
        body = json.loads(request.content)
        if body["think"]:
            return httpx.Response(
                200, json={"response": "{}", "thinking": "not an artifact", "done_reason": "length"}
            )
        case = next(c for c in data["cases"] if f"Question: {c['question']}" in body["prompt"])
        answer = {
            "answer": case["answers"][0] if case["answers"] else "",
            "citations": case["evidence_ids"],
            "abstain": case["abstain"],
        }
        return httpx.Response(
            200,
            json={
                "response": json.dumps(answer),
                "thinking": "not an artifact",
                "done_reason": "stop",
            },
        )

    client = OllamaClient("http://local")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        run_reasoning(client, "local", tmp_path)
    finally:
        client.close()
    saved = (tmp_path / "reasoning.json").read_text()
    report = json.loads(saved)
    assert report["summary"][0]["success_rate"] == 1
    assert report["summary"][1]["success_rate"] == 0
    assert report["summary"][1]["cases"] == 6
    assert report["summary"][1]["length_stops"] == 6
    assert "not an artifact" not in saved
