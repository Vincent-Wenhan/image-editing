"""Growth uses current capacity, verified measurements, configuration and headroom."""

import copy

import pytest
import yaml

from src.core.config import ProjectSettings
from src.core.io import sha256
from src.training.memory import plan_batch


@pytest.fixture
def measured_debug(tmp_path):
    settings = ProjectSettings.load()
    config = copy.deepcopy(settings.recipe)
    config["config"]["process"][0]["train"].update(steps=20, batch_size=2)
    path = tmp_path / "debug.yaml"
    path.write_text(yaml.safe_dump(config))
    report = dict(
        status="complete",
        config=str(path),
        config_sha256=sha256(path),
        peak_reserved_gib=15,
        gpu_total_gib=48,
    )
    requested = copy.deepcopy(config)
    requested["config"]["process"][0]["train"].update(steps=150, batch_size=4)
    policy = settings.experiment["acceptance"]["memory_policy"] | {"minimum_reference_steps": 20}
    return requested, report, policy


@pytest.mark.parametrize("total,used,expected", [(48, 0, 4), (24, 0, 2), (48, 16, 2)])
def test_growth_is_limited_by_actual_leased_card(measured_debug, total, used, expected):
    config, reference, policy = measured_debug
    card = dict(total=total * 1024, used=used * 1024, uuid="GPU-current")
    plan = plan_batch(config, reference, card, policy)
    assert plan["effective_batch_size"] == expected
    assert plan["estimated_peak_gib"] + plan["headroom_gib"] <= plan["available_gib"]


def test_missing_telemetry_prevents_growth_and_small_card_is_rejected(measured_debug):
    config, reference, policy = measured_debug
    reference.pop("gpu_total_gib")
    card = dict(total=48 * 1024, used=0, uuid="GPU-current")
    assert plan_batch(config, reference, card, policy)["effective_batch_size"] == 2
    with pytest.raises(MemoryError, match="headroom"):
        plan_batch(config, reference, card | {"total": 16 * 1024}, policy)


def test_training_parameters_or_reference_config_changes_are_rejected(measured_debug):
    config, reference, policy = measured_debug
    card = dict(total=48 * 1024, used=0, uuid="GPU-current")
    changed = copy.deepcopy(config)
    changed["config"]["process"][0]["datasets"][0]["resolution"] = [1024]
    with pytest.raises(ValueError, match="parameters/data differ"):
        plan_batch(changed, reference, card, policy)
    from pathlib import Path

    Path(reference["config"]).write_text("changed")
    with pytest.raises(ValueError, match="unchanged config"):
        plan_batch(config, reference, card, policy)
