from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from apps.benchmark.dataset_retrieval import (
    QueryResult,
    RunProfile,
    aggregate_results,
    score_ranking,
    select_questions,
)
from apps.benchmark.datasets.memory_agent_bench import (
    adapt_row,
    prepare,
    sha256_file,
    word_chunks,
)
from apps.benchmark.datasets.schema import RetrievalSample, load_samples


def _row() -> dict[str, Any]:
    return {
        "context": "checkout-api timeout is 2 seconds. Backups run weekly.",
        "questions": ["What is the checkout timeout?", "Does checkout use retries?"],
        "answers": [["2 seconds", "2 seconds"], ["yes"]],
        "metadata": {"source": "ruler_test", "qa_pair_ids": ["q0", "q1"]},
    }


def test_word_chunks_preserve_overlap_without_redundant_tail() -> None:
    assert list(word_chunks("a b c d e f g", 4, 1)) == ["a b c d", "d e f g"]
    assert list(word_chunks("a b c d e f g h", 4, 1)) == ["a b c d", "d e f g", "g h"]
    for words, overlap in ((0, 0), (4, -1), (4, 4)):
        with pytest.raises(ValueError):
            list(word_chunks("context", words, overlap))


def test_proxy_labels_use_original_context_and_exclude_ambiguous_answers() -> None:
    sample = adapt_row(_row(), 0)
    answer, ambiguous = sample.questions
    assert answer.relevant_ids == (sample.chunks[0].chunk_id,)
    assert answer.answers == ("2 seconds",)
    assert answer.original_question_id == "q0"
    assert answer.label_method == "answer_span_proxy"
    assert ambiguous.skip_reason == "ambiguous_or_empty_answer"
    assert "Does checkout use retries?" not in sample.chunks[0].text
    assert "yes" not in sample.chunks[0].text


def test_proxy_matches_token_boundaries_and_normalizes_unicode_and_punctuation() -> None:
    row = _row()
    row.update(
        context="The catalog lists Café-Paris.",
        questions=["Where?", "Which animal?"],
        answers=[["Ｃａｆé Paris"], ["cat"]],
    )
    sample = adapt_row(row, 0)
    assert sample.questions[0].relevant_ids
    assert sample.questions[1].skip_reason == "no_answer_span_in_context"


def test_provided_labels_use_full_context_and_question_specific_flags() -> None:
    relevant = {"role": "user", "content": "checkout timeout is 2 seconds"}
    distractor = {"role": "assistant", "content": "Backups happen weekly"}
    row = _row()
    row["context"] = repr(["Chat Time: 2023/01/01", [relevant, distractor]])
    row["answers"] = [["two seconds"], ["weekly"]]
    row["metadata"]["source"] = "longmemeval_test"
    row["metadata"]["haystack_sessions"] = [
        [[{**relevant, "has_answer": True}]],
        [[{**distractor, "has_answer": True}]],
    ]
    sample = adapt_row(row, 17, chunk_words=4, overlap_words=1)
    assert any("Backups happen weekly" in chunk.text for chunk in sample.chunks)
    assert all("has_answer" not in chunk.text for chunk in sample.chunks)
    assert sample.questions[0].label_method == "provided_turn_labels"
    assert len(sample.questions[0].relevant_ids) == 2  # Both children inherit turn relevance.
    assert len(sample.questions[1].relevant_ids) == 1
    assert not set(sample.questions[0].relevant_ids) & set(sample.questions[1].relevant_ids)


def test_missing_provided_evidence_excludes_the_whole_question() -> None:
    present = {"role": "user", "content": "checkout timeout is 2 seconds"}
    missing = {"role": "user", "content": "Other required evidence"}
    row = _row()
    row["context"] = repr(["Chat Time: 2023/01/01", [present]])
    row["metadata"]["haystack_sessions"] = [
        [[{**present, "has_answer": True}, {**missing, "has_answer": True}]],
        [[{**present, "has_answer": False}]],
    ]
    sample = adapt_row(row, 17)
    assert sample.questions[0].relevant_ids == ()
    assert sample.questions[0].skip_reason == "evidence_not_in_context"
    assert sample.questions[1].skip_reason == "no_provided_evidence"


def test_adapter_rejects_misaligned_ground_truth() -> None:
    row = _row()
    row["answers"] = [["2 seconds"]]
    with pytest.raises(ValueError, match="aligned"):
        adapt_row(row, 0)


def test_schema_rejects_unknown_relevance_ids_and_duplicate_chunks() -> None:
    sample = adapt_row(_row(), 0)
    data = sample.model_dump()
    data["questions"][0]["relevant_ids"] = ("not-in-corpus",)
    with pytest.raises(ValueError, match="unknown chunks"):
        RetrievalSample.model_validate(data)
    data = sample.model_dump()
    data["chunks"] = (*data["chunks"], data["chunks"][0])
    with pytest.raises(ValueError, match="duplicate chunk"):
        RetrievalSample.model_validate(data)


def test_prepare_uses_hugging_face_libraries_and_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = pytest.importorskip("datasets")
    hub = pytest.importorskip("huggingface_hub")
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"mock-parquet")
    config = {
        "dataset_id": "ai-hyz/MemoryAgentBench",
        "revision": "a" * 40,
        "split": "Accurate_Retrieval",
        "filename": "data/Accurate_Retrieval.parquet",
        "sha256": sha256_file(raw),
        "chunk_words": 128,
        "overlap_words": 32,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    download = Mock(return_value=str(raw))
    load = Mock(return_value=[_row()])
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(datasets, "load_dataset", load)

    output = tmp_path / "prepared"
    manifest = prepare(config_path, output)

    assert download.call_args.kwargs["repo_id"] == config["dataset_id"]
    assert download.call_args.kwargs["repo_type"] == "dataset"
    assert download.call_args.kwargs["revision"] == config["revision"]
    assert load.call_args.args == ("parquet",)
    assert load.call_args.kwargs["split"] == config["split"]
    assert manifest.sample_count == 1
    assert manifest.question_count == 2
    assert manifest.scored_question_count == 1
    assert manifest.samples_sha256 == sha256_file(output / "samples.jsonl")
    assert load_samples(output / "samples.jsonl")[0].source == "ruler_test"

    config["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    load.reset_mock()
    with pytest.raises(ValueError, match="checksum"):
        prepare(config_path, output, raw_file=raw)
    load.assert_not_called()


def test_sampling_is_stable_and_does_not_hide_unlabeled_questions() -> None:
    sample = adapt_row(_row(), 0)
    other = adapt_row(_row(), 1)
    profile = RunProfile(
        strategies=("semantic",), max_samples_per_source=1, max_queries_per_sample=2
    )
    assert select_questions([sample, other], profile) == {sample.sample_id: sample.questions}
    assert select_questions([sample, other], profile) == select_questions([sample], profile)
    assert any(q.skip_reason for q in select_questions([sample], profile)[sample.sample_id])


def test_cutoff_scores_are_hand_computed_and_mrr_is_bounded_at_k() -> None:
    one, two, five = score_ranking(["noise", "answer"], ["answer", "missing"], (1, 2, 5))
    assert one.precision == one.recall == one.reciprocal_rank == one.ndcg == 0
    assert two.precision == two.recall == two.reciprocal_rank == 0.5
    assert two.ndcg == pytest.approx(0.38685280723454163)
    assert five.precision == 0.2
    assert five.recall == five.reciprocal_rank == 0.5
    with pytest.raises(ValueError, match="unlabeled"):
        score_ranking([], [], (1,))
    with pytest.raises(ValueError, match="duplicate"):
        score_ranking(["answer", "answer"], ["answer"], (2,))


def test_aggregation_weights_queries_equally_and_separates_label_methods() -> None:
    first = QueryResult(
        sample_id="s",
        source="source",
        question_id="q1",
        original_question_id=None,
        question="query",
        label_method="answer_span_proxy",
        strategy="semantic",
        relevant_ids=("a",),
        ranked_ids=("a",),
        quality=score_ranking(["a"], ["a"], (1,)),
        latencies_ms=(2.0, 4.0),
        fallbacks=0,
        candidates_reranked=0,
    )
    second = replace(
        first,
        question_id="q2",
        ranked_ids=(),
        quality=score_ranking([], ["a"], (1,)),
        latencies_ms=(6.0,),
    )
    provided = replace(first, label_method="provided_turn_labels")
    aggregates = aggregate_results([first, second, provided])
    proxy = next(a for a in aggregates if a.label_method == "answer_span_proxy")
    assert proxy.queries == 2
    assert proxy.precision == proxy.recall == proxy.mrr == proxy.ndcg == 0.5
    assert proxy.latency_p50_ms == 4.0
    assert len(aggregates) == 2


def test_profile_rejects_invalid_cutoffs_and_candidate_limits() -> None:
    for kwargs in ({"ks": ()}, {"ks": (5, 1)}, {"ks": (0,)}, {"candidate_limit": 5}):
        with pytest.raises(ValueError):
            RunProfile.model_validate(kwargs)
