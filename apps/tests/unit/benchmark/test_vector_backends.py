from pathlib import Path
from uuid import uuid4

import pytest

from apps.memory_service.persistence.pgvector_index import normalized_vector
from apps.memory_service.persistence.turbovec_index import TurboVecIndex


@pytest.mark.parametrize("values", [[0.0] * 384, [1.0], [float("nan")] * 384, [float("inf")] * 384])
def test_vector_validation_rejects_invalid_vectors(values: list[float]) -> None:
    with pytest.raises(ValueError):
        normalized_vector(values)


def test_checkpoint_roundtrip_and_single_owner(tmp_path: Path) -> None:
    pytest.importorskip("turbovec")
    path = tmp_path / "index.zip"
    index = TurboVecIndex(path)
    tenant, memory = uuid4(), uuid4()
    try:
        index.upsert(42, tenant, memory, 3, [2.0, *([0.0] * 383)])
        index.checkpoint()
        with pytest.raises(RuntimeError, match="live owner"):
            TurboVecIndex(path)
    finally:
        index.close()
    index = TurboVecIndex(path)
    try:
        assert index.entries[42].memory_id == str(memory)
        assert index.entries[42].revision == 3
        index.upsert(42, tenant, memory, 4, [0.0, 1.0, *([0.0] * 382)])
        assert len(index.native) == 1
        assert index.entries[42].revision == 4
        index.remove(42)
        index.remove(42)
        assert not index.entries
    finally:
        index.close()
