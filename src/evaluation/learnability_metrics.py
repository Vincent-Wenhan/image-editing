"""Same-noise edit contrasts and preregistered learnability gates, not human success rates."""

import numpy as np


def contrast_metrics(reference, prediction, identity):
    truth = reference["target"] - reference["source"]
    delta = prediction - identity
    result = {}
    for name, label in [("t", 0), ("d", 1)]:
        selected = (reference["labels"] == label) & reference["valid"]
        gt, predicted = truth[selected], delta[selected]
        magnitude = float(np.mean(np.abs(gt))) if len(gt) else 0
        square = float(np.sum(gt.astype(np.float64) ** 2))
        result[f"{name}_contrast_mae"] = (
            float(np.mean(np.abs(predicted - gt))) / magnitude if magnitude else None
        )
        result[f"{name}_contrast_response"] = (
            float(np.sum(predicted.astype(np.float64) * gt)) / square if square else None
        )
    return result


def learnability_gates(methods, interventions, gate):
    full, original = methods["full_role"]["nonzero"], methods["full_role"]

    def finite(value):
        return value is not None and np.isfinite(value)

    def less(left, right, slack=0):
        return bool(finite(left) and finite(right) and left <= right + slack)

    def improved(left, right, fraction):
        return less(left, right * (1 - fraction) if finite(right) else None)

    fit = dict(
        t=less(full["t_target_mae"], gate["t_max"])
        and improved(
            full["t_target_mae"],
            full["identity_t_target_mae"],
            gate["source_relative_improvement"],
        ),
        d=less(full["d_target_mae"], gate["d_max"])
        and improved(
            full["d_target_mae"],
            full["identity_d_target_mae"],
            gate["source_relative_improvement"],
        ),
        p=less(full["p_source_drift"], gate["p_max"]),
        identity=less(original["by_strength"]["0.0"]["target_mae"], gate["identity_max"]),
        t_contrast=less(full["t_contrast_mae"], gate["contrast_max"]),
        d_contrast=less(full["d_contrast_mae"], gate["contrast_max"]),
    )
    usage = {}
    for name, other in [
        ("constant", methods["constant"]),
        ("zero", interventions["zero"]),
        ("shuffle", interventions["shuffle"]),
    ]:
        region = other["nonzero"]
        usage[name] = dict(
            t=improved(full["t_target_mae"], region["t_target_mae"], gate["role_improvement"]),
            d=improved(full["d_target_mae"], region["d_target_mae"], gate["role_improvement"]),
            p=less(full["p_source_drift"], region["p_source_drift"], gate["identity_tolerance"]),
            identity=less(
                original["by_strength"]["0.0"]["target_mae"],
                other["by_strength"]["0.0"]["target_mae"],
                gate["identity_tolerance"],
            ),
        )
    old = methods["old_new"]["nonzero"]
    d_semantics = dict(
        d=improved(full["d_target_mae"], old["d_target_mae"], gate["role_improvement"]),
        t=improved(full["t_target_mae"], old["t_target_mae"], -gate["role_improvement"]),
        p=less(full["p_source_drift"], old["p_source_drift"], gate["identity_tolerance"]),
    )
    fit_pass = all(fit.values())
    usage_pass = all(all(v.values()) for v in usage.values())
    return dict(
        fit=fit,
        role_usage=usage,
        d_increment=d_semantics,
        fit_pass=fit_pass,
        role_usage_pass=usage_pass,
        d_increment_pass=all(d_semantics.values()),
        passed=fit_pass and usage_pass and all(d_semantics.values()),
    )
