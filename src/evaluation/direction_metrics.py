"""Signed flow endpoint diagnostics; targets are scoring data, never model inputs."""

import numpy as np

BRANCHES = ("positive", "negative", "cfg_native", "cfg_fp32")
REGIONS = ("t", "d", "p")


def bf16_float(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint16:
        raise TypeError("Velocity archives require actual BF16 uint16 bits")
    result = (bits.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite archived BF16 velocity")
    return result


def velocities(archive, guidance):
    result = {branch: bf16_float(archive[branch]) for branch in BRANCHES[:3]}
    p, n = result["positive"], result["negative"]
    result["cfg_fp32"] = n + guidance * (p - n)
    return result


def inner(a, b, weight):
    a, b, weight = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(weight)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or weight.shape != a.shape[:1]
        or not all(np.isfinite(x).all() for x in [a, b, weight])
        or (weight < 0).any()
        or (weight > 1).any()
    ):
        raise ValueError("Direction metrics require aligned finite NxC values and unit areas")
    return (
        float((a * b * weight[:, None]).sum() / (weight.sum() * a.shape[1]))
        if weight.sum()
        else None
    )


def direction(delta, desired, weight, floor):
    energy, power = inner(desired, desired, weight), inner(delta, delta, weight)
    if energy is None:
        return dict(cosine=None, projection=None, contrast_gain=None, delta_rms=None, energy=None)
    dot = inner(delta, desired, weight)
    return dict(
        cosine=dot / np.sqrt(power * energy) if power > floor and energy > floor else None,
        projection=dot / energy if energy > floor else None,
        contrast_gain=(2 * dot - power) / energy if energy > floor else None,
        delta_rms=np.sqrt(power),
        energy=energy,
    )


def score(velocity, identity, zero, original, drop_d, x, anchor, target, masks, time, floor):
    """Negative dt makes anchor-target the correct editing velocity sign."""
    if not 0 < time <= 1 or masks.shape != (3, x.shape[0]):
        raise ValueError("Direction scoring requires a positive model time and T/D/P areas")
    desired = (anchor.astype(np.float64) - target) / time
    endpoint = x.astype(np.float64) - time * velocity
    zero_endpoint = x.astype(np.float64) - time * zero
    original_endpoint = x.astype(np.float64) - time * original
    drop_endpoint = x.astype(np.float64) - time * drop_d
    result = {}
    for name, weight in zip(REGIONS, masks, strict=True):
        reference = anchor if name == "p" else target
        error = inner(endpoint - reference, endpoint - reference, weight)
        baseline = inner(zero_endpoint - reference, zero_endpoint - reference, weight)
        causal = direction(original - drop_d, desired, weight, floor)
        energy = inner(anchor - target, anchor - target, weight)
        drop_error = inner(drop_endpoint - reference, drop_endpoint - reference, weight)
        original_error = inner(original_endpoint - reference, original_endpoint - reference, weight)
        result[name] = dict(
            **direction(velocity - identity, desired, weight, floor),
            endpoint_mse=error,
            zero_endpoint_mse=baseline,
            conditional_gain=(baseline - error) / baseline
            if baseline is not None and baseline > floor
            else None,
            zero_direction=direction(velocity - zero, desired, weight, floor),
            drop_d_direction=causal,
            drop_d_gain=(drop_error - original_error) / energy
            if energy is not None and energy > floor
            else None,
        )
    return result


def cfg_effect(positive_delta, negative_delta, native_delta, fp32_delta, floor):
    weights = np.ones(positive_delta.shape[0])
    p = inner(positive_delta, positive_delta, weights)
    n = inner(negative_delta, negative_delta, weights)
    dot = inner(positive_delta, negative_delta, weights)
    native = inner(native_delta, native_delta, weights)
    quant = native_delta - fp32_delta
    return dict(
        positive_negative_cosine=dot / np.sqrt(p * n) if p > floor and n > floor else None,
        native_rms=np.sqrt(native),
        fp32_rms=np.sqrt(inner(fp32_delta, fp32_delta, weights)),
        cfg_rounding_delta_rms=np.sqrt(inner(quant, quant, weights)),
    )
