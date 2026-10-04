"""Run retrieval diagnostics on a prepared independent dataset; write JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from apps.benchmark.database_sandbox import benchmark_database
from apps.benchmark.dataset_retrieval import RunProfile, format_report, run_dataset
from apps.benchmark.datasets.memory_agent_bench import sha256_file
from apps.benchmark.datasets.schema import DatasetManifest, load_samples

DEFAULT_PROFILE = Path(__file__).parent / "datasets/configs/memory_agent_bench_retrieval.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/benchmarks/memory_agent_bench")
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-prefix", type=Path, required=True, help="writes .json and .md")
    parser.add_argument("--database-url", help="defaults to DATABASE_URL/local PostgreSQL")
    args = parser.parse_args()
    profile = RunProfile.model_validate_json(args.profile.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate_json(
        (args.dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    samples_path = args.dataset_dir / "samples.jsonl"
    if sha256_file(samples_path) != manifest.samples_sha256:
        parser.error("normalized dataset checksum does not match manifest; prepare it again")
    samples = load_samples(samples_path)
    if len(samples) != manifest.sample_count:
        parser.error("sample count does not match manifest")
    from dotenv import load_dotenv

    load_dotenv()
    import torch

    from apps.memory_service.embeddings.cross_encoder import SentenceTransformerCrossEncoder
    from apps.memory_service.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )
    from apps.memory_service.retrieval.reranker import CrossEncoderReranker

    torch.set_num_threads(profile.torch_threads)
    embedder = SentenceTransformerEmbeddingModel(profile.embedding_model, device=profile.device)
    reranker = (
        CrossEncoderReranker(
            SentenceTransformerCrossEncoder(profile.reranking_model, device=profile.device),
            max_candidates=profile.candidate_limit,
        )
        if "hybrid_reranked" in profile.strategies
        else None
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sentence_transformers": version("sentence-transformers"),
        "datasets": version("datasets"),
        "huggingface_hub": version("huggingface-hub"),
        "device": profile.device,
        "torch_threads": str(torch.get_num_threads()),
    }
    with benchmark_database(args.database_url) as uow_factory:
        report = run_dataset(
            uow_factory,
            embedder,
            samples,
            manifest,
            profile,
            reranker=reranker,
            environment=environment,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(format_report(report), encoding="utf-8")
    print(f"Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
