"""Persistent PostgreSQL store and actual LangMem APIs, deterministic model only."""

from contextlib import nullcontext

import pytest
from sqlalchemy import text

from apps.benchmark.run_langmem_comparison import (
    ScriptedSmokeModel,
    persistent_langmem_store,
    run_comparison,
)
from apps.memory_service.embeddings.base import FakeEmbeddingModel

pytest.importorskip("langmem")
pytest.importorskip("langgraph.store.postgres")


def test_three_way_persistent_replay_and_isolated_cleanup(engine, uow_factory):
    url = engine.url.render_as_string(hide_password=False)
    embedder = FakeEmbeddingModel()
    with persistent_langmem_store(url, embedder) as store:
        report = run_comparison(uow_factory, store, ScriptedSmokeModel(), embedder=embedder)
        assert report["sessions"] == 8
        assert report["store_type"] == "PostgresStore"
        assert report["mode"] == "scripted_smoke"
        assert report["tenant_isolation_passed"]
        assert set(report["backends"]) == {
            "custom_memory_os",
            "langmem_hot_path",
            "langmem_background",
        }
        for name, data in report["backends"].items():
            summary = data["summary"]
            assert summary["write_attempts"] == 11
            assert summary["settled_probe_count"] == 19
            assert summary["tenant_leak_queries"] == 0
            assert summary["visible_write_latency_mean_ms"] > 0
            assert len(data["reads"]) == 38
            assert all(r["context_tokens"] <= 160 for r in data["reads"])
            assert data["forget"][0]["deleted_ids"]
            if name == "custom_memory_os":
                assert summary["poison_acceptance"] == 0
                assert summary["stale_fact_rate"] == 0
            else:
                assert summary["poison_acceptance"] == 1
                assert summary["duplicate_rate"] > 0
            if name == "langmem_background":
                assert summary["background_completed"] == 11
                assert summary["background_completion_delay_mean_ms"] > 0
    with engine.connect() as connection:
        assert (
            connection.scalar(
                text("SELECT count(*) FROM pg_namespace WHERE nspname LIKE 'langmem_bench_%'")
            )
            == 0
        )


@pytest.mark.parametrize("fail", [False, True])
def test_isolated_store_persists_across_connections_and_cleans_on_failure(engine, fail):
    import psycopg
    from langgraph.store.postgres import PostgresStore
    from psycopg.conninfo import make_conninfo

    url = engine.url.render_as_string(hide_password=False)
    with (
        pytest.raises(RuntimeError) if fail else nullcontext(),
        persistent_langmem_store(url, FakeEmbeddingModel()) as store,
    ):
        store.put(("persistence-probe",), "key", {"content": {"text": "durable"}})
        with engine.connect() as conn:
            schema = conn.scalar(
                text("SELECT nspname FROM pg_namespace WHERE nspname LIKE 'langmem_bench_%'")
            )
        db = engine.url
        conninfo = make_conninfo(
            host=db.host,
            port=db.port,
            dbname=db.database,
            user=db.username,
            password=db.password,
            options=f"-csearch_path={schema},public",
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            reopened = PostgresStore(conn)
            assert reopened.get(("persistence-probe",), "key").value["content"]["text"] == "durable"
        if fail:
            raise RuntimeError("injected failure")
    with engine.connect() as connection:
        assert (
            connection.scalar(
                text("SELECT count(*) FROM pg_namespace WHERE nspname LIKE 'langmem_bench_%'")
            )
            == 0
        )
