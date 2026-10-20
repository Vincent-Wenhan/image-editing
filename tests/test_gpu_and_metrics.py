"""Ensure occupied GPUs are excluded and identity cannot imply editing success."""

import numpy as np
import pytest

from src.core.random import sample_seed
from src.evaluation.metrics import metrics
from src.infra.gpu import is_idle, require_lease


@pytest.mark.parametrize("change", [dict(process=True), dict(used=129), dict(utilization=6)])
def test_busy_gpu_excluded(change):
    card = dict(process=False, used=0, utilization=0)
    assert is_idle(card)
    assert not is_idle(card | change)


def test_gpu_requires_launcher(monkeypatch):
    monkeypatch.delenv("IMAGE_EDIT_GPU_UUID", raising=False)
    with pytest.raises(RuntimeError, match="image-edit gpu"):
        require_lease()


def test_identity_reports_unfulfilled_target():
    source = np.zeros((2, 2, 3))
    target = np.ones_like(source)
    result = metrics(source, target, source, np.zeros((2, 2)), 0.05)
    assert result["source_drift_mae"] == 0
    assert result["changed_area_fraction"] == 0
    assert result["target_mae"] == result["identity_target_mae"] == 1


def test_seed_is_independent_of_order():
    before = {name: sample_seed(42, name) for name in ["a", "b"]}
    after = {name: sample_seed(42, name) for name in ["b", "a"]}
    assert before == after
    assert before["a"] != before["b"]


def test_launcher_does_not_start_task_on_locked_gpu(tmp_path, monkeypatch):
    import argparse
    import fcntl
    import os
    import sys

    from src.infra import gpu

    card = dict(
        index=0, uuid="GPU-test", name="test", total=48000, process=False, used=0, utilization=0
    )
    monkeypatch.setattr(gpu, "inventory", lambda: [card])
    monkeypatch.setattr(gpu.tempfile, "gettempdir", lambda: str(tmp_path))
    root = tmp_path / f"image-edit-gpu-{os.getuid()}"
    root.mkdir()
    marker = tmp_path / "should-not-exist"
    args = argparse.Namespace(
        device=None, command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    )
    with (root / "GPU-test.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="No idle, unlocked"):
            gpu.run(args)
    assert not marker.exists()


def test_launcher_rechecks_idleness_after_lock(tmp_path, monkeypatch):
    import argparse

    from src.infra import gpu

    card = dict(
        index=0, uuid="GPU-test", name="test", total=48000, process=False, used=0, utilization=0
    )
    responses = iter([[card], [card | {"process": True}]])
    monkeypatch.setattr(gpu, "inventory", lambda: next(responses))
    monkeypatch.setattr(gpu.tempfile, "gettempdir", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="No idle, unlocked"):
        gpu.run(argparse.Namespace(device=None, command=["must-not-run"]))
