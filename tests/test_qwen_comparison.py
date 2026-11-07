"""Matched Qwen comparisons must reject real training/inference budget differences."""

import copy
from pathlib import Path

import pytest

from src.qwen.comparison import _fairness


def methods():
    method = {
        "data_root": Path("same-data"),
        "split": "train",
        "rows": {"sample": {"instruction": "move right"}},
        "inference": {"seed": 42, "steps": 20, "seed_policy": "per_sample"},
        "train_budget": {"rank": 8, "evaluated_updates": 120, "initial_sha256": "same"},
        "train_trace": [{"sample_ids": ["sample"], "noisy_sha256": "same-noise"}],
    }
    return [method, copy.deepcopy(method)]


@pytest.mark.parametrize("change", ["noise", "step", "seed_policy", "split", "initialization"])
def test_qwen_fairness_rejects_mismatched_actual_experiments(change):
    pair = methods()
    if change == "noise":
        pair[1]["train_trace"][0]["noisy_sha256"] = "different-noise"
    elif change == "step":
        pair[1]["train_budget"]["evaluated_updates"] = 20
    elif change == "initialization":
        pair[1]["train_budget"]["initial_sha256"] = "different-init"
    elif change == "split":
        pair[1]["split"] = "dev"
    else:
        pair[1]["inference"]["seed_policy"] = "stream"
    with pytest.raises(ValueError):
        _fairness(pair)


def test_qwen_fairness_accepts_identical_actual_budget_and_noise():
    assert _fairness(methods())["samples"] == 1
