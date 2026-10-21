"""Paired ordinal summaries with crossed reviewer/scene cluster bootstrap.

Scenes have equal weight; repeated strengths and multiple raters do not become
independent observations. Intervals are descriptive, without multiplicity tests.
"""

from itertools import combinations

import numpy as np

DIMENSIONS = ("edit_success", "preservation")


def paired_estimate(
    cells: dict[tuple[str, str], list[float]],
    *,
    seed: int,
    resamples: int,
    inference_ready: bool,
) -> dict:
    """cells[(scene, reviewer)] holds within-rater, within-sample differences."""
    scenes = sorted({scene for scene, _ in cells})
    reviewers = sorted({reviewer for _, reviewer in cells})
    matrix = np.full((len(scenes), len(reviewers)), np.nan)
    for i, scene in enumerate(scenes):
        for j, reviewer in enumerate(reviewers):
            values = cells.get((scene, reviewer), [])
            if values:
                matrix[i, j] = np.mean(values)

    def mean(value):
        # Empty scene/rater combinations remain missing, never zero judgements.
        count = np.isfinite(value).sum(axis=1)
        means = np.divide(np.nansum(value, axis=1), count, where=count > 0)
        return float(means[count > 0].mean()) if (count > 0).any() else None

    estimate = mean(matrix) if scenes else None
    interval = None
    if inference_ready and len(scenes) >= 2 and len(reviewers) >= 3:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(resamples):
            selected = matrix[
                rng.integers(len(scenes), size=len(scenes))[:, None],
                rng.integers(len(reviewers), size=len(reviewers))[None, :],
            ]
            value = mean(selected)
            if value is not None:
                samples.append(value)
        if samples:
            interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return dict(
        mean_difference=estimate,
        ci95=interval,
        scenes=len(scenes),
        reviewers=len(reviewers),
        paired_ratings=sum(len(values) for values in cells.values()),
    )


def summarize(
    judgements: list[dict], methods: list[str], *, ready: bool, seed: int, resamples: int
) -> dict:
    summaries = {}
    for method in methods:
        rows = [row for row in judgements if row["method"] == method]
        entry = dict(rated=len(rows), failures={})
        for dimension in DIMENSIONS:
            scores = [int(row[dimension]) for row in rows if row[dimension] != "NA"]
            entry[dimension] = dict(
                judgeable=len(scores),
                mean=float(np.mean(scores)) if scores else None,
                full_score_fraction=sum(x == 2 for x in scores) / len(scores) if scores else None,
            )
        for row in rows:
            if row["failure"]:
                key = row["failure"]
                entry["failures"][key] = entry["failures"].get(key, 0) + 1
        summaries[method] = entry
    paired = []
    lookup = {(r["reviewer"], r["sample_id"], r["method"]): r for r in judgements}
    for left, right in combinations(methods, 2):
        entry = dict(left=left, right=right, direction="left minus right")
        for dimension in DIMENSIONS:
            cells = {}
            for (reviewer, sample, method), row in lookup.items():
                other = lookup.get((reviewer, sample, right))
                if method != left or other is None or "NA" in (row[dimension], other[dimension]):
                    continue
                key = (row["scene_group"], reviewer)
                cells.setdefault(key, []).append(int(row[dimension]) - int(other[dimension]))
            entry[dimension] = paired_estimate(
                cells, seed=seed, resamples=resamples, inference_ready=ready
            )
        paired.append(entry)
    return dict(methods=summaries, paired=paired)
