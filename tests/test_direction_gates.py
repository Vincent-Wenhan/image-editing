"""Positive output differences alone cannot authorize a timestep-only training comparison."""

import copy

from src.core.config import load_yaml
from src.core.paths import CONFIG_ROOT
from src.evaluation.direction_gates import average, slice_gates


def rows_fixture():
    rows = []
    for group in range(4):
        for method in ["constant", "binary", "old_new", "full_role"]:
            for variant in ["original", "zero", "shuffle", "swap_old_new", "swap_dp", "drop_d"]:
                for time in [1.0, 0.9, 0.5, 0.1]:
                    for strength in [0.0, 1.0]:
                        gain = dict(constant=0.0, binary=0.01, old_new=0.04, full_role=0.08)[method]
                        if time >= 0.9:
                            gain -= 0.03
                        if variant != "original":
                            gain -= 0.02
                        if variant == "zero":
                            gain = 0.0
                        row = dict(
                            id=f"{group}_{strength}",
                            group_id=f"g{group}",
                            split="dev",
                            method=method,
                            variant=variant,
                            time=time,
                            state_family="source_bridge",
                            strength=strength,
                            p_endpoint_mse=0.02 if variant == "zero" else 0.0201,
                        )
                        for region in ["t", "d"]:
                            row.update(
                                {
                                    f"{region}_contrast_gain": gain,
                                    f"{region}_cosine": 0.3,
                                    f"{region}_projection": 0.2,
                                    f"{region}_drop_d_cosine": 0.3,
                                    f"{region}_drop_d_projection": 0.1,
                                    f"{region}_drop_d_gain": 0.02,
                                }
                            )
                        rows.append(row)
    return rows


def test_only_complete_consistent_direction_and_causal_evidence_passes():
    gate = load_yaml(CONFIG_ROOT / "experiments/role_direction.yaml")["gate"]
    rows = rows_fixture()
    assert slice_gates(rows, gate, "dev", 42)["passed"]
    for problem in [
        "wrong_direction",
        "missing_direction",
        "no_d_increment",
        "no_time_gap",
        "p_damage",
    ]:
        broken = copy.deepcopy(rows)
        for row in broken:
            if row["method"] != "full_role":
                continue
            if problem == "wrong_direction":
                row["t_cosine"] = -0.5
            if problem == "missing_direction":
                row["t_cosine"] = None
            if problem == "no_d_increment":
                row["d_drop_d_projection"] = 0.0
            if problem == "no_time_gap" and row["time"] >= 0.9:
                row["d_contrast_gain"] += 0.03
            if problem == "p_damage" and row["variant"] == "original":
                row["p_endpoint_mse"] = 0.03
        assert not slice_gates(broken, gate, "dev", 42)["passed"], problem


def test_scene_average_avoids_strength_pseudoreplication():
    rows = [
        dict(group_id="a", metric=1.0),
        dict(group_id="a", metric=1.0),
        dict(group_id="b", metric=0.0),
    ]
    assert average(rows, "metric") == 0.5
    assert average([], "metric") is None
