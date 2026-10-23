"""Calibrated synthetic translation/shadow proxies, never human semantic judgements."""

import numpy as np


def correlation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid masked-template correlation via FFT, without an additional dependency."""
    h, w = kernel.shape
    height, width = image.shape
    if min(h, w) < 1 or h > height or w > width:
        raise ValueError("Template must fit the image")
    shape = height + h - 1, width + w - 1
    result = np.fft.irfftn(
        np.fft.rfftn(image, s=shape) * np.fft.rfftn(kernel[::-1, ::-1], s=shape), s=shape
    )
    return result[h - 1 : height, w - 1 : width]


def centroid(mask: np.ndarray) -> np.ndarray:
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Occupancy cannot be empty")
    return np.array([x.mean(), y.mean()])


class TranslationLocator:
    """Source template calibrated on known source/target; abstains on weak separation.

    Empty-background/GT calibration is evaluation supervision, never an inference
    condition. Candidate counts are template detections, not reliable object counts.
    """

    def __init__(self, source, target, background, old, new):
        if source.shape != target.shape or source.shape != background.shape:
            raise ValueError("Locator image geometry differs")
        if (
            source.ndim != 3
            or source.shape[-1] != 3
            or any(a.shape != source.shape[:2] or a.dtype != np.bool_ for a in [old, new])
            or any(
                not np.isfinite(a).all() or a.min() < 0 or a.max() > 1
                for a in [source, target, background]
            )
        ):
            raise ValueError("Locator requires aligned boolean occupancy and finite unit RGB")
        self.old, self.new = centroid(old), centroid(new)
        y, x = np.where(old)
        y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
        self.template = source[y0:y1, x0:x1].astype(np.float64)
        self.mask = old[y0:y1, x0:x1].astype(np.float64)
        self.background = np.asarray(background, dtype=np.float64)
        self.offset = self.old - [x0, y0]
        self.extent = float(np.hypot(y1 - y0, x1 - x0))
        self.radius = max(y1 - y0, x1 - x0) * 0.6
        positives, negatives, errors = [], [], []
        for image, expected in [(source, self.old), (target, self.new)]:
            scores = self.scores(image)
            iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
            position = np.array([ix, iy]) + self.offset
            errors.append(float(np.linalg.norm(position - expected)))
            positives.append(float(scores[iy, ix]))
            yy, xx = np.indices(scores.shape)
            far = (yy - iy) ** 2 + (xx - ix) ** 2 > self.radius**2
            negatives.append(float(scores[far].max()) if far.any() else 0.0)
        positive, negative = min(positives), max(negatives + [0.0])
        self.threshold = (positive + negative) / 2
        self.calibration = dict(
            status="calibrated"
            if positive - negative > max(1e-6, positive * 0.1) and max(errors) <= self.extent * 0.1
            else "abstain",
            reference_errors_px=errors,
            positive_min=positive,
            negative_max=negative,
            threshold=self.threshold,
            object_extent_px=self.extent,
            old_centroid=self.old.tolist(),
            new_centroid=self.new.tolist(),
            scope="source-template and GT/background calibration; synthetic-family proxy",
        )

    def scores(self, prediction):
        image = np.asarray(prediction, dtype=np.float64)
        if image.shape != self.background.shape or not np.isfinite(image).all():
            raise ValueError("Prediction must match finite calibration geometry")
        area = self.mask.sum() * 3
        bg_error = correlation(((image - self.background) ** 2).sum(-1), self.mask)
        obj_error = correlation((image**2).sum(-1), self.mask)
        for channel in range(3):
            obj_error -= 2 * correlation(
                image[..., channel], self.template[..., channel] * self.mask
            )
        obj_error += float((self.template**2 * self.mask[..., None]).sum())
        return (bg_error - obj_error) / area

    def locate(self, prediction) -> dict:
        empty = dict(
            status="abstain",
            candidates=[],
            position_error_px=None,
            position_error_extents=None,
            distance_to_old_px=None,
            template_duplicate_candidate=None,
        )
        if self.calibration["status"] != "calibrated":
            return dict(empty, reason="reference_calibration_failed")
        scores = self.scores(prediction)
        yy, xx = np.indices(scores.shape)
        peaks = []
        for _ in range(3):
            y, x = np.unravel_index(np.argmax(scores), scores.shape)
            score = float(scores[y, x])
            if score < self.threshold:
                break
            position = np.array([x, y]) + self.offset
            peaks.append(dict(x=float(position[0]), y=float(position[1]), score=score))
            scores[(yy - y) ** 2 + (xx - x) ** 2 <= self.radius**2] = -np.inf
        if not peaks:
            return dict(empty, reason="no_confident_template")
        first = np.array([peaks[0]["x"], peaks[0]["y"]])
        error = float(np.linalg.norm(first - self.new))
        return dict(
            status="localized",
            candidates=peaks,
            position_error_px=error,
            position_error_extents=error / self.extent,
            distance_to_old_px=float(np.linalg.norm(first - self.old)),
            template_duplicate_candidate=len(peaks) > 1,
        )


def change_proxies(source, target, prediction, background, labels, valid) -> dict:
    """Separate creation/removal direction on causal D and edge preservation on P.

    Dark-pixel IoU is restricted to known D, and can be confounded by foreground
    intrusion. It is not an automatic judgement that a physical shadow is correct.
    """
    if (
        source.shape != target.shape
        or source.shape != prediction.shape
        or source.shape != background.shape
        or labels.shape != source.shape[:2]
        or valid.shape != labels.shape
    ):
        raise ValueError("Scene diagnostic geometry differs")
    if not all(np.isfinite(x).all() for x in [source, target, prediction, background]):
        raise ValueError("Non-finite scene diagnostic pixels")
    d, p = (labels == 1) & valid, (labels == 2) & valid
    weights = np.array([0.2126, 0.7152, 0.0722])
    s, t, pred, bg = [x @ weights for x in [source, target, prediction, background]]

    def error(mask):
        return float(np.abs(prediction - target)[mask].mean()) if mask.any() else None

    delta, actual = target - source, prediction - source
    denominator = float(np.square(delta[d]).sum())
    directed = float((delta[d] * actual[d]).sum() / denominator) if denominator else None
    truth_dark, pred_dark = (bg - t > bg * 0.5) & d, (bg - pred > bg * 0.5) & d
    union = truth_dark | pred_dark
    edge_sum, edge_count = 0.0, 0
    for axis in (0, 1):
        paired = (p[1:] & p[:-1]) if axis == 0 else (p[:, 1:] & p[:, :-1])
        difference = np.abs(np.diff(prediction, axis=axis) - np.diff(source, axis=axis))
        edge_sum += float(difference[paired].sum())
        edge_count += int(paired.sum()) * 3
    return dict(
        d_shadow_removal_mae=error(d & (t - s > 1 / 255)),
        d_shadow_creation_mae=error(d & (s - t > 1 / 255)),
        d_signed_change_response=directed,
        d_dark_region_iou=float((truth_dark & pred_dark).sum() / union.sum())
        if union.any()
        else None,
        p_edge_mae=edge_sum / edge_count if edge_count else None,
    )
