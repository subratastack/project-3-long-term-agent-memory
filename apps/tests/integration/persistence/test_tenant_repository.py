"""Integration tests for `TenantRepository` against real PostgreSQL.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).
"""

from collections.abc import Callable
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from apps.memory_service.domain.models import Tenant
from apps.memory_service.persistence.unit_of_work import UnitOfWork

UowFactory = Callable[[], UnitOfWork]


def _name() -> str:
    return f"tenant-{uuid4()}"


def test_a_tenant_round_trips(uow_factory: UowFactory) -> None:
    tenant = Tenant(name=_name(), description="support bot memories")
    with uow_factory() as uow:
        uow.tenants.add(tenant)
        uow.commit()

    with uow_factory() as uow:
        by_id = uow.tenants.get(tenant.tenant_id)
        by_name = uow.tenants.get_by_name(tenant.name)

    assert by_id == tenant
    assert by_name == tenant


def test_unknown_tenants_are_none(uow_factory: UowFactory) -> None:
    with uow_factory() as uow:
        assert uow.tenants.get(uuid4()) is None
        assert uow.tenants.get_by_name(_name()) is None


def test_names_are_unique(uow_factory: UowFactory) -> None:
    name = _name()
    with uow_factory() as uow:
        uow.tenants.add(Tenant(name=name))
        uow.commit()

    with pytest.raises(IntegrityError), uow_factory() as uow:
        uow.tenants.add(Tenant(name=name))
        uow.commit()


def test_list_all_includes_every_registered_tenant(uow_factory: UowFactory) -> None:
    tenants = [Tenant(name=_name()) for _ in range(3)]
    with uow_factory() as uow:
        for tenant in tenants:
            uow.tenants.add(tenant)
        uow.commit()

    with uow_factory() as uow:
        listed = uow.tenants.list_all()

    assert {t.tenant_id for t in tenants} <= {t.tenant_id for t in listed}
