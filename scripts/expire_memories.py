"""Run once from cron/systemd: expire, decay and compact one tenant or all tenants.

Usage: python scripts/expire_memories.py --tenant-id UUID
       python scripts/expire_memories.py --all-tenants

DATABASE_URL selects the database. Emits changed IDs as JSON. Re-running in the
same daily decay interval is a no-op. No records are hard-deleted or automatically
tombstoned. Schedule this command in the deployment's own maintenance system.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from apps.memory_service.consolidation.forgetting import maintain_memories
from apps.memory_service.persistence.unit_of_work import (
    UnitOfWork,
    build_engine,
    build_session_factory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--tenant-id", type=UUID)
    scope.add_argument("--all-tenants", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    engine = build_engine()
    factory = build_session_factory(engine)
    now = datetime.now(UTC)
    try:
        with UnitOfWork(factory) as uow:
            # Includes legacy records whose tenant has not been registered.
            if args.all_tenants:
                from sqlalchemy import select

                from apps.memory_service.persistence.models import MemoryRecordRow

                tenants = list(uow.session.scalars(select(MemoryRecordRow.tenant_id).distinct()))
            else:
                tenants = [args.tenant_id]
        for tenant_id in sorted(tenants, key=str):
            report = maintain_memories(lambda: UnitOfWork(factory), tenant_id, now=now)
            print(json.dumps({"tenant_id": str(tenant_id), **report}, default=str, sort_keys=True))
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
