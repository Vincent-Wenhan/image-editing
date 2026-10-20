"""Ensure occupied GPUs are excluded and identity cannot imply editing success."""

import numpy as np
import pytest

from image_editing.common import sample_seed
from image_editing.evaluation import metrics
from image_editing.gpu import is_idle, require_lease


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
