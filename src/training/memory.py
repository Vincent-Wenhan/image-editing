"""Conservative batch planning from a verified debug run and the current leased GPU."""

import copy
import math
from pathlib import Path

from src.core.config import load_yaml
from src.core.io import sha256


def comparable_process(configuration: dict) -> dict:
    process = copy.deepcopy(configuration["config"]["process"][0])
    for key in ("steps", "batch_size"):
        process["train"].pop(key, None)
    process["save"].pop("save_every", None)
    return process


def plan_batch(configuration: dict, reference: dict, card: dict, policy: dict) -> dict:
    reference_path = Path(reference["config"])
    if (
        reference.get("status") != "complete"
        or sha256(reference_path) != reference["config_sha256"]
    ):
        raise ValueError("Batch reference must be a complete run with an unchanged config")
    previous = load_yaml(reference_path)
    if comparable_process(previous) != comparable_process(configuration):
        raise ValueError("Batch reference training parameters/data differ from the requested run")
    baseline = previous["config"]["process"][0]["train"]["batch_size"]
    requested = configuration["config"]["process"][0]["train"]["batch_size"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (baseline, requested)
    ):
        raise ValueError("Batch sizes must be positive integers")
    peak = reference.get("peak_reserved_gib")
    if not isinstance(peak, int | float) or not math.isfinite(peak) or peak <= 0:
        raise ValueError("Batch reference has no valid reserved-memory measurement")
    fraction, minimum = policy["headroom_fraction"], policy["minimum_headroom_gib"]
    growth = policy["max_growth_factor"]
    if not 0 < fraction < 1 or minimum <= 0 or not 1 <= growth <= 2:
        raise ValueError("Invalid batch memory policy")
    total, available = card["total"] / 1024, (card["total"] - card["used"]) / 1024
    reserve = max(minimum, total * fraction)
    if peak + reserve > available:
        raise MemoryError("Current leased GPU lacks headroom even for the measured debug batch")
    eligible = (
        previous["config"]["process"][0]["train"]["steps"] >= policy["minimum_reference_steps"]
        and reference.get("gpu_total_gib", 0) > 0
        and not reference.get("telemetry_errors")
    )
    chosen = min(requested, max(1, int(baseline * growth)) if eligible else baseline)
    while chosen > baseline and peak * (chosen / baseline) + reserve > available:
        chosen -= 1
    estimate = peak * max(1, chosen / baseline)
    return dict(
        requested_batch_size=requested,
        baseline_batch_size=baseline,
        effective_batch_size=chosen,
        estimated_peak_gib=estimate,
        available_gib=available,
        total_gib=total,
        headroom_gib=reserve,
        gpu_uuid=card["uuid"],
        reason="capacity_bound" if eligible else "insufficient_growth_evidence",
    )
