"""Single-owner TurboVec index with an atomic vector + UUID/revision checkpoint."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.memory_service.indexing.index_status import VectorIndexState
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS, MemoryRecordRow
from apps.memory_service.persistence.pgvector_index import eligibility, normalized_vector
from apps.memory_service.persistence.vector_repository import VectorMatch
from apps.memory_service.retrieval.filters import RetrievalFilters


@dataclass(frozen=True)
class IndexEntry:
    tenant_id: str
    memory_id: str
    revision: int
    vector_sha256: str


def vector_digest(vector: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes()).hexdigest()


class TurboVecIndex:
    """One writer/reader process per checkpoint; use PostgreSQL to recover any loss."""

    def __init__(self, path: Path, *, bit_width: int = 4) -> None:
        from turbovec import IdMapIndex  # type: ignore[import-untyped]

        if bit_width not in (2, 3, 4):
            raise ValueError("bit_width must be 2, 3, or 4")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._owner = path.with_suffix(path.suffix + ".lock").open("a+b")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise RuntimeError("TurboVec checkpoint already has a live owner") from None
        self.native: Any = IdMapIndex(dim=EMBEDDING_DIMENSIONS, bit_width=bit_width)
        self.entries: dict[int, IndexEntry] = {}
        try:
            if path.exists():
                with zipfile.ZipFile(path) as checkpoint:
                    manifest = json.loads(checkpoint.read("manifest.json"))
                    if manifest["version"] != 1 or manifest["bit_width"] != bit_width:
                        raise ValueError("checkpoint version/bit width mismatch")
                    self.native = IdMapIndex.from_bytes(checkpoint.read("index.tvim"))
                    self.entries = {int(k): IndexEntry(**v) for k, v in manifest["entries"].items()}
                if self.native.dim != EMBEDDING_DIMENSIONS or len(self.native) != len(self.entries):
                    raise ValueError("checkpoint inventory does not match vectors")
                if any(key not in self.native for key in self.entries):
                    raise ValueError("checkpoint missing a manifest vector")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._owner.close()

    def upsert(
        self,
        vector_id: int,
        tenant_id: UUID,
        memory_id: UUID,
        revision: int,
        vector: Sequence[float],
    ) -> None:
        normalized = normalized_vector(vector)
        with self.lock:
            self.native.remove(vector_id)
            # If add fails after remove, remove the manifest too; reconciliation can retry.
            self.entries.pop(vector_id, None)
            self.native.add_with_ids(
                np.asarray([normalized], dtype=np.float32), np.asarray([vector_id], dtype=np.uint64)
            )
            self.entries[vector_id] = IndexEntry(
                str(tenant_id), str(memory_id), revision, vector_digest(vector)
            )

    def remove(self, vector_id: int) -> None:
        with self.lock:
            self.native.remove(vector_id)
            self.entries.pop(vector_id, None)

    def checkpoint(self) -> None:
        """fsync both payloads in one file, rename atomically, then fsync the directory."""
        with self.lock:
            fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=".turbovec-")
            try:
                with os.fdopen(fd, "w+b") as stream:
                    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as checkpoint:
                        checkpoint.writestr("index.tvim", self.native.to_bytes())
                        checkpoint.writestr(
                            "manifest.json",
                            json.dumps(
                                {
                                    "version": 1,
                                    "bit_width": self.native.bit_width,
                                    "entries": {str(k): asdict(v) for k, v in self.entries.items()},
                                },
                                sort_keys=True,
                            ),
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(name):
                    os.unlink(name)

    def find_nearest(
        self,
        session: Session,
        filters: RetrievalFilters,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorMatch]:
        if limit < 1:
            raise ValueError("limit must be positive")
        vector = normalized_vector(query_embedding)
        state = VectorIndexState
        # SQL authorization is paid on every read; pending/stale vectors are excluded.
        rows = session.execute(
            select(state.vector_id, state.memory_id, state.revision)
            .join(
                MemoryRecordRow,
                (state.tenant_id == MemoryRecordRow.tenant_id)
                & (state.memory_id == MemoryRecordRow.memory_id),
            )
            .where(
                *eligibility(filters),
                state.state == "indexed",
                state.indexed_revision == state.revision,
            )
        )
        with self.lock:
            expected_tenant = str(filters.tenant_id)
            allowed = []
            # Scalar projections are fresh SQL reads; no ORM identity-cache hydration.
            for vector_id, memory_id, revision in rows:
                entry = self.entries.get(vector_id)
                if entry is not None and (
                    entry.tenant_id == expected_tenant
                    and entry.memory_id == str(memory_id)
                    and entry.revision == revision
                ):
                    allowed.append(vector_id)
            if not allowed:
                return []
            scores, ids = self.native.search(
                np.asarray([vector], dtype=np.float32),
                k=limit,
                allowlist=np.asarray(allowed, dtype=np.uint64),
            )
            return [
                VectorMatch(UUID(self.entries[int(key)].memory_id), 1 - float(score))
                for key, score in zip(ids[0], scores[0], strict=True)
            ]
