from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord, Provenance, TemporalValidity

NOW = datetime(2026, 9, 28, tzinfo=UTC)


@pytest.fixture
def episodes():
    tenant = uuid4()

    def make(count=2, **changes):
        records = []
        for i in range(count):
            observed = NOW - timedelta(days=20) + timedelta(hours=i)
            values = dict(
                tenant_id=tenant,
                content="pool exhaustion",
                confidence=0.9,
                provenance=[
                    Provenance(
                        event_id=uuid4(),
                        source_type=SourceType.SYSTEM_EVENT,
                        source_reference=f"incident-{i}",
                        observed_at=observed,
                        trust_level=TrustLevel.SYSTEM,
                    )
                ],
                temporal_validity=TemporalValidity(valid_from=observed),
                memory_type=MemoryType.EPISODIC,
                status=MemoryStatus.ACTIVE,
                trust_level=TrustLevel.SYSTEM,
                subject_keys=["service:payments"],
                metadata={"category_key": "connection-pool exhaustion"},
            )
            values.update(changes)
            records.append(MemoryRecord(**values))
        return records

    return make
