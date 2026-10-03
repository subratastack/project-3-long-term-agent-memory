"""Native Store probes without PostgreSQL or model downloads."""

from dataclasses import replace
from uuid import uuid4

import pytest

from apps.memory_service.embeddings.base import FakeEmbeddingModel
from integrations.langgraph.store_comparison import (
    LangGraphStoreExperiment,
    capability_rows,
    comparison_dataset,
    comparison_probes,
)


@pytest.fixture
def native():
    return LangGraphStoreExperiment(FakeEmbeddingModel(), {"A": uuid4(), "B": uuid4()}, uuid4())


def test_shared_fixture_covers_the_requested_cases():
    dataset = comparison_dataset()
    assert len(dataset) == 7
    assert len({entry.label for entry in dataset}) == 7
    assert {entry.tenant for entry in dataset} == {"A", "B"}
    assert sum(entry.poisoned for entry in dataset) == 1
    assert {e.label for e in dataset if e.label.startswith("incident_a")} == {
        "incident_a1",
        "incident_a2",
    }
    assert len(capability_rows()) == 8


def test_native_namespaces_separate_tenants_but_are_not_authorization(native):
    for entry in comparison_dataset():
        native.write(entry)
    probe_a = comparison_probes()[3]
    probe_b = comparison_probes()[4]
    assert "incident_b1" not in native.read(probe_a, 160).retrieved
    assert native.read(probe_b, 160).retrieved == ("incident_b1",)
    namespaces = {item.namespace for item in native.store.search(native.prefix, limit=30)}
    assert native.namespace("A") in namespaces
    assert native.namespace("B") in namespaces


def test_native_store_uses_the_supplied_embedding_index(native):
    dataset = {e.label: e for e in comparison_dataset()}
    native.write(dataset["preference"])
    native.write(dataset["incident_a1"])
    hits = native.store.search(native.namespace("A"), query="concise email updates")
    assert hits[0].key == "preference"
    assert hits[0].score is not None and hits[1].score is not None
    assert hits[0].score > hits[1].score


def test_native_provenance_is_json_and_version_keys_have_no_as_of_policy(native):
    for entry in comparison_dataset():
        native.write(entry)
    item = native.store.get(native.namespace("A"), "timeout_old")
    assert item.value["provenance"][0]["event_id"]
    historic = native.read(comparison_probes()[2], 160)
    assert {"timeout_old", "timeout_new", "poison"}.issubset(historic.retrieved)
    assert historic.temporal_applied is False
    native.store.put(native.namespace("A"), "arbitrary", {"text": "No evidence required."})
    assert native.store.get(native.namespace("A"), "arbitrary").value.get("provenance") is None


def test_native_result_limit_is_not_a_token_budget_and_poison_is_stored(native):
    for entry in comparison_dataset():
        native.write(entry)
    result = native.read(comparison_probes()[1], 1)
    assert result.tokens > result.token_budget
    assert "poison" in result.presented
    assert result.retrieved == result.presented


def test_native_delete_removes_key_and_another_experiment_has_separate_namespaces(native):
    preference = comparison_dataset()[0]
    native.write(preference)
    other = LangGraphStoreExperiment(
        FakeEmbeddingModel(), native.tenant_ids, uuid4(), store=native.store
    )
    other.write(preference)
    native.forget(preference.label, "A", now=preference.observed_at)
    assert native.store.get(native.namespace("A"), preference.label) is None
    assert other.store.get(other.namespace("A"), preference.label) is not None
    assert native.read(replace(comparison_probes()[0], tenant="B"), 160).retrieved == ()
