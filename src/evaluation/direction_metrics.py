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
    """Score all regional weighted products together; preserve the scalar definitions."""
    values = [
        np.asarray(v, np.float64)
        for v in [velocity, identity, zero, original, drop_d, x, anchor, target]
    ]
    velocity, identity, zero, original, drop_d, x, anchor, target = values
    masks = np.asarray(masks, np.float64)
    if (
        not 0 < time <= 1
        or x.ndim != 2
        or masks.shape != (3, x.shape[0])
        or any(v.shape != x.shape or not np.isfinite(v).all() for v in values)
        or not np.isfinite(masks).all()
        or (masks < 0).any()
        or (masks > 1).any()
    ):
        raise ValueError(
            "Direction scoring requires finite aligned velocities, time and unit areas"
        )
    area = masks.sum(1)

    def products(a, b):
        return np.divide(masks @ (a * b).mean(1), area, out=np.zeros(3), where=area > 0)

    desired = (anchor - target) / time
    energy = products(desired, desired)

    def directions(delta):
        power, dot = products(delta, delta), products(delta, desired)
        return [
            dict(
                cosine=float(dot[i] / np.sqrt(power[i] * energy[i]))
                if area[i] > 0 and power[i] > floor and energy[i] > floor
                else None,
                projection=float(dot[i] / energy[i]) if area[i] > 0 and energy[i] > floor else None,
                contrast_gain=float((2 * dot[i] - power[i]) / energy[i])
                if area[i] > 0 and energy[i] > floor
                else None,
                delta_rms=float(np.sqrt(power[i])) if area[i] > 0 else None,
                energy=float(energy[i]) if area[i] > 0 else None,
            )
            for i in range(3)
        ]

    def errors(v):
        endpoint = x - time * v
        result = products(endpoint - target, endpoint - target)
        result[2] = products(endpoint - anchor, endpoint - anchor)[2]
        return result

    error, baseline, original_error, drop_error = [
        errors(v) for v in [velocity, zero, original, drop_d]
    ]
    edit, zero_delta, causal = [
        directions(v) for v in [velocity - identity, velocity - zero, original - drop_d]
    ]
    clean_energy = products(anchor - target, anchor - target)
    return {
        name: dict(
            **edit[i],
            endpoint_mse=float(error[i]) if area[i] > 0 else None,
            zero_endpoint_mse=float(baseline[i]) if area[i] > 0 else None,
            conditional_gain=float((baseline[i] - error[i]) / baseline[i])
            if area[i] > 0 and baseline[i] > floor
            else None,
            zero_direction=zero_delta[i],
            drop_d_direction=causal[i],
            drop_d_gain=float((drop_error[i] - original_error[i]) / clean_energy[i])
            if area[i] > 0 and clean_energy[i] > floor
            else None,
        )
        for i, name in enumerate(REGIONS)
    }


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
