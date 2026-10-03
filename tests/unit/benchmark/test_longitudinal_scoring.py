"""Independent scoring checks for the multi-session fixture."""

from dataclasses import replace

import pytest

from apps.benchmark.run_longitudinal_eval import (
    NoMemoryExperiment,
    Session,
    longitudinal_sessions,
    score_read,
    summarize_backend,
)
from integrations.langgraph.store_comparison import (
    JAN_20,
    ReadObservation,
    TaskProbe,
    comparison_dataset,
)


def test_timeline_reuses_shared_fixture_without_seeding_future_memories():
    sessions = longitudinal_sessions()
    assert len(sessions) == 7
    assert [s.at for s in sessions] == sorted(s.at for s in sessions)
    entries = [entry for s in sessions for entry in s.writes]
    assert len(entries) == 9
    assert {e.label for e in comparison_dataset()} < {e.label for e in entries}
    seen = set()
    for session in sessions:
        seen.update(entry.label for entry in session.writes)
        for probe in session.probes:
            assert probe.required_any <= seen
        assert set(session.temporal_queries) <= {p.key for p in session.probes}
        assert set(session.forgetting_queries) <= {p.key for p in session.probes}
    assert sum(len(s.probes) for s in sessions) == 17
    assert sum(len(s.temporal_queries) for s in sessions) == 6
    assert sum(len(s.forgetting_queries) for s in sessions) == 2


@pytest.mark.parametrize(
    "shown, tokens, expected",
    [
        (("preference",), 10, True),
        (("incident_a1",), 10, False),
        (("preference", "poison"), 10, False),
        (("preference", "incident_b1"), 10, False),
        (("preference",), 101, False),
        ((), 0, False),
    ],
)
def test_success_requires_recall_safety_scope_and_budget(shown, tokens, expected):
    probe = TaskProbe("preference", "A", "email updates", JAN_20, frozenset({"preference"}))
    session = Session("score", JAN_20, (), (probe,))
    observation = ReadObservation(
        "test", probe.key, "A", JAN_20.isoformat(), shown, shown, "text", tokens, 100, True
    )
    tenants = {e.label: e.tenant for e in comparison_dataset()}
    score = score_read(session, probe, observation, tenants, poison_arrived=True)
    assert score.task_passed is expected


def test_temporal_and_forgetting_checks_have_explicit_denominators():
    session = longitudinal_sessions()[1]
    probe = session.probes[1]  # Empty B contains forbidden A labels, not a forgetting check.
    read = NoMemoryExperiment().read(probe, 160)
    score = score_read(session, probe, read, {}, poison_arrived=False)
    assert not score.temporal_probe
    assert not score.forgetting_probe
    session = longitudinal_sessions()[-1]
    probe = session.probes[1]
    read = NoMemoryExperiment().read(probe, 160)
    score = score_read(session, probe, read, {}, poison_arrived=True)
    assert score.temporal_probe
    assert score.forgetting_probe


def test_no_memory_does_not_get_recall_credit_for_empty_answers():
    participant = NoMemoryExperiment()
    sessions = longitudinal_sessions()
    reads = tuple(
        score_read(s, p, participant.read(p, 160), {}, poison_arrived=s.at >= sessions[4].at)
        for s in sessions
        for p in s.probes
    )
    summary = summarize_backend(participant.name, reads, (), {})
    assert summary.recall_tasks == 13
    assert summary.recalled_tasks == 0
    assert summary.passed_tasks == 4
    assert summary.poison_opportunities == 8
    assert summary.temporal_probes == 6
    assert summary.forgetting_probes == 2


def test_forbidden_current_version_does_not_receive_history_credit():
    session = longitudinal_sessions()[2]
    probe = session.probes[1]
    read = ReadObservation(
        "test",
        probe.key,
        "A",
        probe.as_of.isoformat(),
        ("timeout_old", "timeout_new"),
        ("timeout_old", "timeout_new"),
        "text",
        30,
        160,
        False,
    )
    score = score_read(
        session, probe, read, {"timeout_old": "A", "timeout_new": "A"}, poison_arrived=False
    )
    assert score.recalled is True
    assert score.task_passed is False
    assert score.forbidden_presented == ("timeout_new",)
    safe_read = replace(read, presented=("timeout_old",))
    assert (
        score_read(
            session, probe, safe_read, {"timeout_old": "A"}, poison_arrived=False
        ).task_passed
        is True
    )
