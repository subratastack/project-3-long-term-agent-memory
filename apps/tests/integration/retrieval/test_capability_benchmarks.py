"""Exercise the new labeled reports through real PostgreSQL service paths."""

import json
from pathlib import Path

from apps.benchmark.capabilities.system_tracks import (
    run_conflicts,
    run_forgetting,
    run_write_policy,
)
from apps.memory_service.ingestion.service import UowFactory


def test_system_reports_preserve_expected_labels_and_lifecycle_guarantees(
    uow_factory: UowFactory,
    tmp_path: Path,
) -> None:
    run_write_policy(uow_factory, tmp_path)
    run_conflicts(uow_factory, tmp_path)
    run_forgetting(uow_factory, tmp_path)
    policy = json.loads((tmp_path / "write_policy.json").read_text())["summary"]
    conflicts = json.loads((tmp_path / "conflict_resolution.json").read_text())["summary"]
    forgetting = json.loads((tmp_path / "forgetting.json").read_text())["summary"]
    assert policy["poison_accepted"] == 0
    assert policy["benign_accepted"] == policy["benign_attempts"]
    assert policy["reason_code_coverage"] == 1
    assert conflicts["visible_set_accuracy"] == 1
    assert forgetting["pass_rate"] == 1
    assert forgetting["checks"] == 17
