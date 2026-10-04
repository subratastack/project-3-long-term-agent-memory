"""Validated, dataset-independent input to retrieval benchmark runners."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalChunk(DatasetModel):
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)


class RetrievalQuestion(DatasetModel):
    question_id: str = Field(min_length=1)
    original_question_id: str | None = None
    text: str = Field(min_length=1)
    answers: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    label_method: str = Field(min_length=1)
    skip_reason: str | None = None

    @model_validator(mode="after")
    def check_labels(self) -> RetrievalQuestion:
        if len(set(self.relevant_ids)) != len(self.relevant_ids):
            raise ValueError("duplicate relevant IDs")
        if bool(self.relevant_ids) == bool(self.skip_reason):
            raise ValueError("provide relevant IDs or an explicit skip reason")
        return self


class RetrievalSample(DatasetModel):
    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    context_chars: int = Field(ge=1)
    chunks: tuple[RetrievalChunk, ...] = Field(min_length=1)
    questions: tuple[RetrievalQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> RetrievalSample:
        chunk_ids = {chunk.chunk_id for chunk in self.chunks}
        if len(chunk_ids) != len(self.chunks):
            raise ValueError("duplicate chunk IDs")
        question_ids = {question.question_id for question in self.questions}
        if len(question_ids) != len(self.questions):
            raise ValueError("duplicate question IDs")
        if any(not set(q.relevant_ids) <= chunk_ids for q in self.questions):
            raise ValueError("question references unknown chunks")
        return self


class DatasetManifest(DatasetModel):
    schema_version: Literal["retrieval-dataset-v1"] = "retrieval-dataset-v1"
    adapter_version: str
    dataset_id: str
    revision: str
    split: str
    source_url: str
    raw_sha256: str
    samples_sha256: str
    chunk_words: int = Field(gt=0)
    overlap_words: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    question_count: int = Field(ge=1)
    scored_question_count: int = Field(ge=0)


def load_samples(path: Path) -> list[RetrievalSample]:
    """Read JSONL produced by any adapter; reject malformed input before running."""
    with path.open(encoding="utf-8") as stream:
        samples = [RetrievalSample.model_validate_json(line) for line in stream if line.strip()]
    if not samples:
        raise ValueError("dataset has no samples")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("duplicate sample IDs")
    return samples
