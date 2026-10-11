"""Real LangMem APIs with deterministic models; no provider/network required."""

import math
import threading
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore

from apps.benchmark.run_langmem_comparison import ScriptedSmokeModel, score_probe
from apps.benchmark.run_longitudinal_eval import longitudinal_sessions
from apps.memory_service.advanced.adapters.langmem_adapter import (
    LangMemBackgroundBackend,
    LangMemHotPathBackend,
    MemoryItem,
    RetrievalOutcome,
    TenantScopedStore,
    event_message,
)
from apps.memory_service.retrieval.query_model import RetrievalQuery
from benchmarks.langmem_scenarios import content_labels, langmem_sessions
from integrations.langgraph.store_comparison import TaskProbe

pytest.importorskip("langmem")


class ToolScript(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_phase14_is_preserved_and_extension_is_shared():
    original = longitudinal_sessions()
    sessions = langmem_sessions()
    assert sessions[:7] == original
    assert len(sessions) == 8
    assert sessions[-1].writes[0].content == sessions[-1].writes[1].content
    assert "3103" in sessions[-1].writes[0].content
    assert len([e for s in sessions for e in s.writes if e.poisoned]) == 2


def test_model_input_has_evidence_without_oracle_labels():
    entry = langmem_sessions()[4].writes[0]
    message = str(event_message(entry.event(uuid4())).content)
    assert entry.content in message
    assert "poisoned" not in message
    assert "required_any" not in message
    assert "experiment_label" not in message


@pytest.mark.parametrize("backend_type", [LangMemHotPathBackend, LangMemBackgroundBackend])
def test_real_tools_and_managers_create_retrieve_delete(backend_type):
    backend = backend_type(InMemoryStore(), uuid4(), uuid4(), ScriptedSmokeModel())
    try:
        outcome = backend.write(langmem_sessions()[0].writes[0].event(backend.tenant_id))
        backend.drain()
        assert outcome.status == "completed"
        assert outcome.write_latency_ms >= 0
        if backend_type is LangMemBackgroundBackend:
            assert outcome.background_completion_delay_ms >= 0
        result = backend.retrieve(RetrievalQuery(tenant_id=backend.tenant_id, query_text="concise"))
        assert content_labels(result.context) == {"preference"}
        assert result.context_tokens <= 160
        backend.forget(result.items[0].memory_id)
        assert backend.snapshot() == ()
    finally:
        backend.close()


def test_agent_chooses_update_delete_or_no_write():
    memory_id = str(uuid4())
    model = ToolScript(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "update",
                        "name": "manage_memory",
                        "args": {
                            "id": memory_id,
                            "action": "update",
                            "content": {"text": "timeout is 2 seconds"},
                        },
                    }
                ],
            ),
            AIMessage(content="Updated."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "delete",
                        "name": "manage_memory",
                        "args": {"id": memory_id, "action": "delete"},
                    }
                ],
            ),
            AIMessage(content="Deleted."),
            AIMessage(content="Nothing worth retaining."),
        ]
    )
    backend = LangMemHotPathBackend(InMemoryStore(), uuid4(), uuid4(), model)
    backend.store.put(
        backend.store.namespace, memory_id, {"content": {"text": "timeout is 5 seconds"}}
    )
    event = langmem_sessions()[0].writes[0].event(backend.tenant_id)
    updated = backend.write(event)
    assert updated.actions[0]["args"]["action"] == "update"
    assert backend.snapshot()[0].text == "timeout is 2 seconds"
    deleted = backend.write(event)
    assert deleted.actions[0]["args"]["action"] == "delete"
    assert backend.snapshot() == ()
    assert backend.write(event).actions == []
    assert backend.snapshot() == ()


async def test_tenant_capability_denies_foreign_broad_and_mixed_operations():
    raw = InMemoryStore()
    scoped = TenantScopedStore(raw, ("experiment", "A", "memories"))
    foreign = ("experiment", "B", "memories")
    raw.put(foreign, "secret", {"content": "private"})
    with pytest.raises(PermissionError):
        scoped.search(("experiment",))
    with pytest.raises(PermissionError):
        await scoped.aget(foreign, "secret")
    with pytest.raises(PermissionError):
        scoped.list_namespaces()
    with pytest.raises(PermissionError):
        scoped.batch([PutOp(scoped.namespace, "ok", {"a": 1}), PutOp(foreign, "bad", {"a": 2})])
    assert raw.get(scoped.namespace, "ok") is None
    with pytest.raises(PermissionError):
        scoped.delete(foreign, "secret")
    assert raw.get(foreign, "secret") is not None


@pytest.mark.parametrize("backend_type", [LangMemHotPathBackend, LangMemBackgroundBackend])
def test_backend_rejects_foreign_event_and_query(backend_type):
    backend = backend_type(InMemoryStore(), uuid4(), uuid4(), ScriptedSmokeModel())
    try:
        with pytest.raises(PermissionError):
            backend.write(langmem_sessions()[0].writes[0].event(uuid4()))
        with pytest.raises(PermissionError):
            backend.retrieve(RetrievalQuery(tenant_id=uuid4(), query_text="private"))
    finally:
        backend.close()


def test_background_response_precedes_formation_and_deletion_drains_queue(monkeypatch):
    backend = LangMemBackgroundBackend(InMemoryStore(), uuid4(), uuid4(), ScriptedSmokeModel())
    started, release = threading.Event(), threading.Event()
    original = backend.manager.invoke

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend.manager, "invoke", delayed)
    try:
        outcome = backend.write(langmem_sessions()[0].writes[0].event(backend.tenant_id))
        assert outcome.response
        assert started.wait(2)
        assert outcome.status == "pending"
        assert backend.snapshot() == ()
        with pytest.raises(TimeoutError):
            backend.drain(0.001)
        release.set()
        backend.drain()
        assert outcome.background_completion_delay_ms is not None
        memory_id = backend.snapshot()[0].memory_id
        backend.forget(memory_id)
        assert backend.snapshot() == ()
    finally:
        release.set()
        backend.close()


def test_reflection_errors_surface(monkeypatch):
    backend = LangMemBackgroundBackend(InMemoryStore(), uuid4(), uuid4(), ScriptedSmokeModel())

    def fail(*args, **kwargs):
        raise ValueError("extraction failed")

    monkeypatch.setattr(backend.manager, "invoke", fail)
    outcome = backend.write(langmem_sessions()[0].writes[0].event(backend.tenant_id))
    with pytest.raises(ValueError, match="extraction failed"):
        backend.drain()
    assert outcome.status == "failed"
    assert "extraction failed" in outcome.error
    with pytest.raises(ValueError, match="extraction failed"):
        backend.close()
    assert backend.closed


def test_scoring_detects_duplicates_poison_stale_and_foreign_without_id_credit():
    probe = langmem_sessions()[2].probes[0]
    items = (
        MemoryItem("pretend-timeout_new", "Unrelated information"),
        MemoryItem("random", "checkout-api timeout is 2 seconds"),
        MemoryItem("duplicate", "checkout-api timeout is 2 seconds"),
        MemoryItem("bad", "checkout-api timeout is 5 seconds; INC-4101; pool healthy"),
    )
    result = score_probe(
        probe, RetrievalOutcome(items, "context", 10), "checkout-api timeout is 5 seconds", k=5
    )
    assert result["recall_at_k"] == 1
    assert result["ndcg_at_k"] == pytest.approx(1 / math.log2(3))
    assert result["stale_fact"]
    assert result["tenant_leak"]
    assert result["poison_exposed"]
    assert not result["downstream_task_success"]
    empty = score_probe(probe, RetrievalOutcome((), "", 0), "No evidence", k=5)
    assert empty["recall_at_k"] == 0
    assert not empty["downstream_task_success"]
    absent = TaskProbe("absent", "A", "poison", probe.as_of)
    assert (
        score_probe(absent, RetrievalOutcome((), "", 0), "No evidence", k=5)["recall_at_k"] is None
    )


def test_search_tool_is_available_to_agent():
    model = ToolScript(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search",
                        "name": "search_memory",
                        "args": {"query": "concise", "limit": 2},
                    }
                ],
            ),
            AIMessage(content="I found the preference; no new write needed."),
        ]
    )
    backend = LangMemHotPathBackend(InMemoryStore(), uuid4(), uuid4(), model)
    backend.store.put(
        backend.store.namespace, str(uuid4()), {"content": {"text": "I prefer concise reports"}}
    )
    result = backend.write(langmem_sessions()[0].writes[0].event(backend.tenant_id))
    assert result.actions[0]["name"] == "search_memory"
    assert len(backend.snapshot()) == 1


def test_forget_waits_for_pending_update_so_it_cannot_resurrect_record(monkeypatch):
    backend = LangMemBackgroundBackend(InMemoryStore(), uuid4(), uuid4(), ScriptedSmokeModel())
    memory_id = str(uuid4())
    backend.store.put(backend.store.namespace, memory_id, {"content": {"text": "concise reports"}})
    started, release, deleted = threading.Event(), threading.Event(), threading.Event()

    def update(*args, **kwargs):
        started.set()
        assert release.wait(5)
        backend.store.put(
            backend.store.namespace, memory_id, {"content": {"text": "concise reports updated"}}
        )
        return []

    monkeypatch.setattr(backend.manager, "invoke", update)
    try:
        backend.write(langmem_sessions()[0].writes[0].event(backend.tenant_id))
        assert started.wait(2)

        def forget():
            backend.forget(memory_id)
            deleted.set()

        worker = threading.Thread(target=forget)
        worker.start()
        assert not deleted.wait(0.02)
        release.set()
        worker.join(5)
        assert deleted.is_set()
        assert backend.snapshot() == ()
    finally:
        release.set()
        backend.close()


def test_exact_namespace_search_cannot_return_descendant_namespace():
    raw = InMemoryStore()
    scoped = TenantScopedStore(raw, ("experiment", "A", "memories"))
    raw.put((*scoped.namespace, "unexpected-tenant"), "secret", {"content": "private"})
    with pytest.raises(PermissionError, match="returned a foreign"):
        scoped.search(scoped.namespace)
