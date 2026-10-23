"""Preregistered evidence gates for a future timestep-only comparison, never a training launch."""

import math

import numpy as np

from src.evaluation.role_metrics import paired_scene_ci


def finite(value):
    return value is not None and math.isfinite(value)


def average(rows, key):
    groups = {}
    for row in rows:
        if finite(row.get(key)):
            groups.setdefault(row["group_id"], []).append(row[key])
    return float(np.mean([np.mean(groups[k]) for k in sorted(groups)])) if groups else None


def collapse_times(rows, metric):
    indexed = {}
    for row in rows:
        indexed.setdefault(row["id"], []).append(row)
    return [
        dict(
            id=k,
            group_id=v[0]["group_id"],
            strength=v[0]["strength"],
            **{metric: average(v, metric)},
        )
        for k, v in sorted(indexed.items())
    ]


def slice_gates(rows, gate, split, seed):
    """One family/training seed/CFG branch; rows must contain the full primary matrix."""
    primary = [r for r in rows if r["state_family"] == "source_bridge" and r["split"] == split]

    def selected(method, variant="original", times=(0.5, 0.1), identity=False):
        return [
            r
            for r in primary
            if r["method"] == method
            and r["variant"] == variant
            and r["time"] in times
            and (r["strength"] == 0 if identity else r["strength"] > 0)
        ]

    def mean(method, metric, variant="original", times=(0.5, 0.1), identity=False):
        return average(selected(method, variant, times, identity), metric)

    def difference(method, key, variant, reference):
        a, b = mean(method, key, variant), mean(method, key, reference)
        return a - b if finite(a) and finite(b) else None

    def at_least(value, minimum):
        return finite(value) and value >= minimum

    directional = {}
    for method, regions in [("old_new", ["t"]), ("full_role", ["t", "d"])]:
        for region in regions:
            part = selected(method)
            values = {
                m: mean(method, f"{region}_{m}") for m in ["cosine", "projection", "contrast_gain"]
            }
            covered = bool(part) and all(
                finite(r.get(f"{region}_{m}")) for r in part for m in values
            )
            directional[f"{method}_{region}"] = covered and all(
                at_least(values[m], gate[f"{m}_min"]) for m in values
            )
    margin = gate["comparison_margin"]
    controls = {
        "old_new_swap_t": at_least(
            difference("old_new", "t_contrast_gain", "original", "swap_old_new"), margin
        )
    }
    for region, variants in [("t", ["swap_old_new", "shuffle"]), ("d", ["swap_dp", "shuffle"])]:
        for variant in variants:
            controls[f"full_{region}_{variant}"] = at_least(
                difference("full_role", f"{region}_contrast_gain", "original", variant), margin
            )
    zero = selected("full_role", "zero")
    controls["zero_contrast_exact"] = bool(zero) and all(
        r.get(f"{k}_contrast_gain") == 0 for r in zero for k in ["t", "d"]
    )
    increments = {}
    for method in ["binary", "old_new"]:
        a, b = mean("full_role", "d_contrast_gain"), mean(method, "d_contrast_gain")
        increments[f"full_over_{method}_d"] = at_least(
            a - b if finite(a) and finite(b) else None, margin
        )
    a, b = mean("full_role", "t_contrast_gain"), mean("old_new", "t_contrast_gain")
    increments["t_not_worse"] = finite(a) and finite(b) and a >= b - margin
    for metric, minimum in [
        ("cosine", gate["cosine_min"]),
        ("projection", gate["projection_min"]),
        ("gain", margin),
    ]:
        increments[f"drop_d_{metric}"] = at_least(mean("full_role", f"d_drop_d_{metric}"), minimum)
    preservation = {}
    for identity in [False, True]:
        a = mean("full_role", "p_endpoint_mse", times=(1.0, 0.9, 0.5, 0.1), identity=identity)
        b = mean(
            "full_role",
            "p_endpoint_mse",
            variant="zero",
            times=(1.0, 0.9, 0.5, 0.1),
            identity=identity,
        )
        preservation["identity" if identity else "p"] = (
            finite(a)
            and finite(b)
            and a - b
            <= max(gate["preservation_relative_max"] * b, gate["preservation_absolute_tolerance"])
        )
    gaps = {}
    for region in ["t", "d"]:
        key = f"{region}_contrast_gain"
        low = collapse_times(selected("full_role"), key)
        high = collapse_times(selected("full_role", times=(1.0, 0.9)), key)
        interval = paired_scene_ci(low, high, key, seed=seed, resamples=gate["bootstrap_resamples"])
        interval["passed"] = at_least(interval["mean_difference"], gate["high_noise_gap_min"]) and (
            split != "dev" or (interval["ci95"] is not None and interval["ci95"][0] > 0)
        )
        gaps[region] = interval
    groups = dict(
        directional=directional, controls=controls, increments=increments, preservation=preservation
    )
    return dict(
        **groups,
        high_noise_gap=gaps,
        passed=all(all(v.values()) for v in groups.values())
        and all(v["passed"] for v in gaps.values()),
    )
