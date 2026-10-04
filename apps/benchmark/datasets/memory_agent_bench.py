"""Prepare pinned MemoryAgentBench Accurate_Retrieval as independent JSONL.

Usage: python -m apps.benchmark.datasets.memory_agent_bench
Only original context is chunked. Answers and evidence tags are scoring data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from apps.benchmark.datasets.schema import (
    DatasetManifest,
    RetrievalChunk,
    RetrievalQuestion,
    RetrievalSample,
)
from apps.memory_service.embeddings.hf_auth import huggingface_token

ADAPTER_VERSION = "memory-agent-bench-v1"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "memory_agent_bench.json"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def word_chunks(text: str, words: int, overlap: int) -> Iterator[str]:
    if words < 1 or not 0 <= overlap < words:
        raise ValueError("chunk words must be positive and overlap within [0, chunk words)")
    tokens = text.split()
    for start in range(0, len(tokens), words - overlap):
        yield " ".join(tokens[start : start + words])
        if start + words >= len(tokens):
            break


def normalize_span(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _turn_key(role: str, content: str) -> str:
    # Hash exact decoded content, not answer text, to align evidence with context.
    return hashlib.sha256(json.dumps([role, content], ensure_ascii=False).encode()).hexdigest()


def adapt_row(
    row: Mapping[str, Any], row_index: int, *, chunk_words: int = 128, overlap_words: int = 32
) -> RetrievalSample:
    """Use supplied turn evidence where available, otherwise an answer-span proxy."""
    context = row["context"]
    questions, answers, metadata = row["questions"], row["answers"], row["metadata"]
    if not isinstance(context, str) or not context.strip():
        raise ValueError("missing context")
    if not questions or len(questions) != len(answers):
        raise ValueError("questions and answers must be nonempty and aligned")
    source = metadata["source"]
    sample_id = f"memory-agent-bench:Accurate_Retrieval:{row_index}"
    chunks: list[RetrievalChunk] = []
    evidence_chunks: dict[str, list[str]] = {}
    haystacks = metadata.get("haystack_sessions")
    if haystacks is not None:
        if len(haystacks) != len(questions):
            raise ValueError("haystack_sessions must align with questions")
        # Upstream stores the full corpus as a literal alternating date/session list.
        # Metadata contains only evidence sessions, so indexing it would drop distractors.
        sessions = ast.literal_eval(context)
        if not isinstance(sessions, list) or len(sessions) % 2:
            raise ValueError("expected alternating date/session context")
        for session_index in range(0, len(sessions), 2):
            date, turns = sessions[session_index : session_index + 2]
            if not isinstance(date, str) or not isinstance(turns, list):
                raise ValueError("invalid session context")
            for turn_index, turn in enumerate(turns):
                role, content = turn["role"], turn["content"]
                key = _turn_key(role, content)
                parent_id = f"{sample_id}:session{session_index // 2}:turn{turn_index}"
                for index, text in enumerate(word_chunks(content, chunk_words, overlap_words)):
                    chunk = RetrievalChunk(
                        chunk_id=f"{parent_id}:chunk{index}",
                        text=f"{date}\n{role}: {text}",
                        parent_id=parent_id,
                    )
                    chunks.append(chunk)
                    evidence_chunks.setdefault(key, []).append(chunk.chunk_id)
    else:
        for index, text in enumerate(word_chunks(context, chunk_words, overlap_words)):
            chunks.append(
                RetrievalChunk(chunk_id=f"{sample_id}:chunk{index}", text=text, parent_id=sample_id)
            )

    normalized_chunks = [(chunk.chunk_id, f" {normalize_span(chunk.text)} ") for chunk in chunks]
    qa_ids = metadata.get("qa_pair_ids")
    if qa_ids is not None and len(qa_ids) != len(questions):
        raise ValueError("qa_pair_ids must align with questions")
    adapted: list[RetrievalQuestion] = []
    for index, (question, aliases) in enumerate(zip(questions, answers, strict=True)):
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError("answers must be lists of strings")
        relevant: set[str] = set()
        method: Literal["provided_turn_labels", "answer_span_proxy"]
        if haystacks is not None:
            method = "provided_turn_labels"
            missing = False
            for session in haystacks[index]:
                for turn in session:
                    if turn["has_answer"]:
                        ids = evidence_chunks.get(_turn_key(turn["role"], turn["content"]), [])
                        missing |= not ids
                        relevant.update(ids)
            reason = "evidence_not_in_context" if missing else "no_provided_evidence"
            if missing:
                relevant.clear()
        else:
            method = "answer_span_proxy"
            spans = {normalize_span(alias) for alias in aliases}
            spans = {
                span
                for span in spans
                if len(span) >= 3 and span not in {"yes", "no", "true", "false", "none", "unknown"}
            }
            for chunk_id, normalized in normalized_chunks:
                if any(f" {span} " in normalized for span in spans):
                    relevant.add(chunk_id)
            reason = "no_answer_span_in_context" if spans else "ambiguous_or_empty_answer"
        adapted.append(
            RetrievalQuestion(
                question_id=f"{sample_id}:question{index}",
                original_question_id=qa_ids[index] if qa_ids is not None else None,
                text=question,
                answers=tuple(dict.fromkeys(aliases)),
                relevant_ids=tuple(sorted(relevant)),
                label_method=method,
                skip_reason=None if relevant else reason,
            )
        )
    return RetrievalSample(
        sample_id=sample_id,
        source=source,
        row_index=row_index,
        context_chars=len(context),
        chunks=tuple(chunks),
        questions=tuple(adapted),
    )


def prepare(config_path: Path, output_dir: Path, raw_file: Path | None = None) -> DatasetManifest:
    """Access the pinned Hub dataset through Hugging Face libraries, then adapt it."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    revision = config["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("dataset revision must be a full commit SHA")
    url = (
        f"https://huggingface.co/datasets/{config['dataset_id']}/resolve/"
        f"{revision}/{config['filename']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("install benchmark dependencies: uv sync --extra benchmark") from exc
    if raw_file is None:
        raw_file = Path(
            hf_hub_download(
                repo_id=config["dataset_id"],
                repo_type="dataset",
                filename=config["filename"],
                revision=revision,
                cache_dir=output_dir / "hf_hub",
                token=huggingface_token(),
            )
        )
    if sha256_file(raw_file) != config["sha256"]:
        raise ValueError("raw file checksum does not match pinned configuration")
    # The Hub download preserves revision/checksum provenance; Datasets supplies
    # the tabular interface, Arrow caching, and an identical offline local-file path.
    dataset = load_dataset(
        "parquet",
        data_files={config["split"]: str(raw_file.resolve())},
        split=config["split"],
        cache_dir=str(output_dir / "hf_datasets"),
    )
    samples_path = output_dir / "samples.jsonl"
    temporary_samples = output_dir / "samples.jsonl.tmp"
    sample_count = question_count = scored_count = 0
    with temporary_samples.open("w", encoding="utf-8") as stream:
        for row_index, row in enumerate(dataset):
            sample = adapt_row(
                row,
                row_index,
                chunk_words=config["chunk_words"],
                overlap_words=config["overlap_words"],
            )
            stream.write(sample.model_dump_json() + "\n")
            sample_count += 1
            question_count += len(sample.questions)
            scored_count += sum(bool(q.relevant_ids) for q in sample.questions)
    temporary_samples.replace(samples_path)
    manifest = DatasetManifest(
        adapter_version=ADAPTER_VERSION,
        dataset_id=config["dataset_id"],
        revision=revision,
        split=config["split"],
        source_url=url,
        raw_sha256=config["sha256"],
        samples_sha256=sha256_file(samples_path),
        chunk_words=config["chunk_words"],
        overlap_words=config["overlap_words"],
        sample_count=sample_count,
        question_count=question_count,
        scored_question_count=scored_count,
    )
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/benchmarks/memory_agent_bench")
    )
    parser.add_argument(
        "--raw-file", type=Path, help="use a checksum-verified existing Parquet file"
    )
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    print(prepare(args.config, args.output_dir, args.raw_file).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
