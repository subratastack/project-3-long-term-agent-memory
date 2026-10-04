"""Run independent memory capability tracks and save Markdown/JSON learning reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from apps.benchmark.capabilities.common import table
from apps.benchmark.capabilities.model_tracks import (
    run_answer_quality,
    run_extraction,
    run_reasoning,
)
from apps.benchmark.capabilities.ollama import OllamaClient
from apps.benchmark.capabilities.system_tracks import (
    run_conflicts,
    run_forgetting,
    run_packing,
    run_write_policy,
)
from apps.benchmark.database_sandbox import benchmark_database

TRACKS = (
    "write_policy",
    "conflict_resolution",
    "forgetting",
    "prompt_packing",
    "answer_quality",
    "memory_extraction",
    "reasoning",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", choices=TRACKS, nargs="+", default=list(TRACKS))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/reports/capabilities-initial")
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--answer-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--extraction-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--reasoning-model", default="qwen3:14b")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_ctx < 4096 or args.num_predict <= 0 or args.timeout <= 0:
        parser.error("provide num-ctx >= 4096, positive num-predict and timeout")
    if len(set(args.tracks)) != len(args.tracks):
        parser.error("tracks must be distinct")
    load_dotenv()
    output = args.output_dir
    system_tracks = set(args.tracks) & set(TRACKS[:4])
    if system_tracks:
        with benchmark_database() as factory:
            if "write_policy" in system_tracks:
                run_write_policy(factory, output)
            if "conflict_resolution" in system_tracks:
                run_conflicts(factory, output)
            if "forgetting" in system_tracks:
                run_forgetting(factory, output)
            if "prompt_packing" in system_tracks:
                import torch

                from apps.memory_service.embeddings.sentence_transformer import (
                    SentenceTransformerEmbeddingModel,
                )

                torch.set_num_threads(4)
                embedder = SentenceTransformerEmbeddingModel(device="cpu")
                run_packing(factory, embedder, output)
        print("Database benchmark transaction rolled back.", flush=True)
    model_tracks = set(args.tracks) & set(TRACKS[4:])
    if model_tracks:
        client = OllamaClient(
            args.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            timeout=args.timeout,
        )
        try:
            if "answer_quality" in model_tracks:
                run_answer_quality(client, args.answer_model, output)
            if "memory_extraction" in model_tracks:
                run_extraction(client, args.extraction_model, output)
            if "reasoning" in model_tracks:
                run_reasoning(client, args.reasoning_model, output)
        finally:
            client.close()
    rows = [
        [track.replace("_", " "), f"[Learning report]({track}.md)", f"[Traces]({track}.json)"]
        for track in TRACKS
        if (output / f"{track}.md").exists()
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(
        "# Memory capability learning reports\n\n"
        "Each report records one benchmark track, its scoring method, measured results, "
        "and limits. Model-backed tracks use installed local Ollama models. "
        "Policy, conflicts, lifecycle, and packing use the existing PostgreSQL database "
        "inside a transaction that is rolled back.\n\n"
        + table(["Track", "Markdown", "JSON"], rows)
        + "\n\n[Method and rerun instructions](../../capability-benchmarks.md)\n"
    )
    print(f"Reports saved under {output}", flush=True)


if __name__ == "__main__":
    main()
