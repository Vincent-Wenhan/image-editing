"""Semantic direction requires the correct sign, magnitude and regional support."""

import numpy as np
import pytest

from src.evaluation.direction_metrics import bf16_float, direction, inner, score, velocities


@pytest.mark.parametrize(
    "scale,projection,gain", [(1, 1, 1), (-1, -1, -3), (0, 0, 0), (0.5, 0.5, 0.75)]
)
def test_flow_direction_perfect_reversed_absent_and_partial(scale, projection, gain):
    desired = np.ones((4, 3))
    result = direction(scale * desired, desired, np.ones(4), 1e-12)
    assert result["projection"] == projection
    assert result["contrast_gain"] == gain
    assert result["cosine"] == (None if scale == 0 else np.sign(scale))


def test_orthogonal_large_output_change_hurts_direction():
    d = np.array([[1.0, 0.0]])
    result = direction(np.array([[0.0, 2.0]]), d, np.ones(1), 1e-12)
    assert result["cosine"] == 0 and result["projection"] == 0 and result["contrast_gain"] == -4
    assert direction(d, d, np.zeros(1), 1e-12)["cosine"] is None
    assert direction(d, np.zeros_like(d), np.ones(1), 1e-12)["contrast_gain"] is None
    with pytest.raises(ValueError):
        inner(d, d, np.ones(2))
    with pytest.raises(ValueError):
        inner(d, d, np.array([-1.0]))


@pytest.mark.parametrize("time", [1.0, 0.9, 0.5, 0.1])
def test_target_seeking_flow_sign_and_denoising_are_distinct(time):
    anchor = np.zeros((6, 2))
    target = anchor.copy()
    target[:2] = 1
    target[2:4] = -0.5
    noise = np.ones_like(anchor) * 0.3
    x = (1 - time) * anchor + time * noise
    # The model sees a source-only x; target is only used to score these candidate velocities.
    source_velocity = (x - anchor) / time
    correct = (x - target) / time
    masks = np.repeat(np.eye(3), 2, axis=1)
    result = score(
        correct,
        source_velocity,
        source_velocity,
        correct,
        source_velocity,
        x,
        anchor,
        target,
        masks,
        time,
        1e-12,
    )
    for region in ["t", "d"]:
        assert result[region]["cosine"] == pytest.approx(1)
        assert result[region]["contrast_gain"] == pytest.approx(1)
        assert result[region]["endpoint_mse"] == pytest.approx(0, abs=1e-15)
        assert result[region]["conditional_gain"] == pytest.approx(1)
    assert result["p"]["endpoint_mse"] == pytest.approx(0, abs=1e-15)
    # A generic perfectly denoised source has no extra edit direction.
    absent = score(
        source_velocity,
        source_velocity,
        source_velocity,
        source_velocity,
        source_velocity,
        x,
        anchor,
        target,
        masks,
        time,
        1e-12,
    )
    assert absent["t"]["projection"] == 0 and absent["d"]["contrast_gain"] == 0


def test_bf16_archive_is_exact_and_not_float16():
    values = np.array([0x3F80, 0xBF80, 0x4000], np.uint16)
    assert np.array_equal(bf16_float(values), [1, -1, 2])
    with pytest.raises(TypeError):
        bf16_float(values.astype(np.float16))
    with pytest.raises(FloatingPointError):
        bf16_float(np.array([0x7F80], np.uint16))
    rows = velocities(dict(positive=values, negative=values, cfg_native=values), 4)
    assert np.array_equal(rows["cfg_fp32"], [1, -1, 2])


def test_native_cfg_preserves_torch_rounding_instead_of_premature_float():
    torch = pytest.importorskip("torch")
    from src.backends.direction_probe import native_cfg

    generator = torch.Generator().manual_seed(42)
    p = torch.randn((2, 8), generator=generator).bfloat16()
    n = torch.randn((2, 8), generator=generator).bfloat16()
    actual = native_cfg(p, n, 4.0)
    assert torch.equal(actual, n + 4.0 * (p - n))
    assert not torch.equal(actual.float(), n.float() + 4.0 * (p.float() - n.float()))
