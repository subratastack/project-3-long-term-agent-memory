from apps.benchmark.dataset_retrieval import RunProfile, run_dataset
from apps.benchmark.datasets.memory_agent_bench import adapt_row
from apps.benchmark.datasets.schema import DatasetManifest
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.ingestion.service import UowFactory


def test_dataset_runner_uses_actual_retrieval_and_keeps_contexts_isolated(
    uow_factory: UowFactory,
) -> None:
    samples = [
        adapt_row(
            {
                "context": f"checkout timeout is {seconds} seconds. Backups happen weekly.",
                "questions": ["checkout", "Does checkout use retries?"],
                "answers": [[f"{seconds} seconds"], ["yes"]],
                "metadata": {"source": "fixture"},
            },
            index,
            chunk_words=6,
            overlap_words=0,
        )
        for index, seconds in enumerate((2, 5))
    ]
    manifest = DatasetManifest(
        adapter_version="fixture-v1",
        dataset_id="test/fixture",
        revision="a" * 40,
        split="Accurate_Retrieval",
        source_url="https://example.org/fixture",
        raw_sha256="0" * 64,
        samples_sha256="0" * 64,
        chunk_words=6,
        overlap_words=0,
        sample_count=2,
        question_count=4,
        scored_question_count=2,
    )
    profile = RunProfile(ks=(1, 2), candidate_limit=2, strategies=("lexical", "semantic", "hybrid"))
    report = run_dataset(uow_factory, FakeEmbeddingModel(), samples, manifest, profile)

    assert len(report.results) == 6
    assert sum(c.scored for c in report.coverage) == 2
    assert sum(c.selected for c in report.coverage) == 4
    for result in report.results:
        assert all(chunk_id.startswith(result.sample_id + ":") for chunk_id in result.ranked_ids)
        assert result.quality[0].precision == 1.0
        assert result.quality[0].recall == 1.0
    assert len(report.construction) == 2
