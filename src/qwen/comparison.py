"""Fair, paired comparison of completed Qwen synthetic Dev scores."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from src.core.config import load_yaml
from src.core.io import read_rows, write_json
from src.evaluation.role_metrics import paired_scene_ci

PAIR_METRICS = (
    "t_target_mae",
    "d_target_mae",
    "p_source_drift",
    "target_mae",
    "changed_fraction",
    "old_only_target_mae",
    "new_only_target_mae",
    "position_error_px",
    "distance_to_old_px",
    "d_shadow_removal_mae",
    "d_shadow_creation_mae",
    "d_signed_change_response",
    "d_dark_region_iou",
    "p_edge_mae",
)


def _paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in result:
            raise ValueError("Methods must use unique LABEL=metrics.json values")
        result[label] = Path(path).resolve()
    if len(result) < 2:
        raise ValueError("At least two scored methods are required")
    return result


def _load(label: str, path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    predictions = Path(metrics["predictions"]).resolve()
    inference = json.loads((predictions / "inference.json").read_text(encoding="utf-8"))
    run_root = Path(metrics["run"]).resolve()
    run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    data_root = Path(run["data_root"]).resolve()
    rows = {row["id"]: row for row in read_rows(data_root / "dev/manifest.csv")}
    items = {item["sample_id"]: item for item in inference.get("results", [])}
    records = {item["id"]: item for item in metrics["records"]}
    if inference.get("status") != "complete" or not (set(items) == set(records) == set(rows)):
        raise ValueError(f"Incomplete or mismatched Qwen cohort: {label}")
    for sample_id, item in items.items():
        row = rows[sample_id]
        if (
            item["source_sha256"] != row["source_sha256"]
            or item["target_sha256"] != row["target_sha256"]
        ):
            raise ValueError(f"Source/target provenance changed: {label}/{sample_id}")
    train_budget = None
    if inference.get("baseline") != "base_source_only":
        config = load_yaml(run["train_config"])
        train_budget = {
            key: config[key] for key in ("max_train_steps", "rank", "train_batch_size", "seed")
        }
    return {
        "label": label,
        "metrics": metrics,
        "inference": inference,
        "run": run,
        "data_root": data_root,
        "rows": rows,
        "records": records,
        "train_budget": train_budget,
    }


def _fairness(methods: list[dict]) -> dict:
    reference = methods[0]
    for method in methods[1:]:
        if method["data_root"] != reference["data_root"]:
            raise ValueError("Compared Qwen methods use different data roots")
        for sample_id in reference["rows"]:
            left, right = reference["rows"][sample_id], method["rows"][sample_id]
            if left["instruction"] != right["instruction"]:
                raise ValueError(f"Prompt differs for {sample_id}")
        if (
            method["inference"]["seed"],
            method["inference"]["steps"],
        ) != (
            reference["inference"]["seed"],
            reference["inference"]["steps"],
        ):
            raise ValueError("Compared Qwen methods use different inference seed or steps")
    budgets = {
        json.dumps(method["train_budget"], sort_keys=True)
        for method in methods
        if method["train_budget"]
    }
    if len(budgets) > 1:
        raise ValueError("Trained Qwen methods use different budgets")
    return {
        "data_root": str(reference["data_root"]),
        "samples": len(reference["rows"]),
        "seed": reference["inference"]["seed"],
        "steps": reference["inference"]["steps"],
        "train_budget": json.loads(next(iter(budgets))) if budgets else None,
        "source_target_prompt_sha_bound": True,
    }


def run(args) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new Qwen comparison directory: {output}")
    methods = [_load(label, path) for label, path in _paths(args.method).items()]
    fairness = _fairness(methods)
    summaries = {}
    for method in methods:
        records = method["records"]
        summaries[method["label"]] = {
            "baseline": method["inference"].get("baseline"),
            "condition": method["metrics"].get("condition"),
            "run": str(method["run"]),
            "predictions": str(Path(method["metrics"]["predictions"]).resolve()),
            "all": method["metrics"]["all"],
            "nonzero": method["metrics"]["nonzero"],
            "semantic": method["metrics"].get("semantic"),
            "records": records,
        }
    pairs = {}
    for left, right in itertools.combinations(methods, 2):
        left_records = [row for row in left["records"].values() if row["strength"] > 0]
        right_records = [row for row in right["records"].values() if row["strength"] > 0]
        pairs[f"{left['label']}__minus__{right['label']}"] = {
            metric: paired_scene_ci(
                left_records, right_records, metric, seed=args.seed, resamples=args.resamples
            )
            for metric in PAIR_METRICS
        }
    output.mkdir(parents=True)
    write_json(
        output / "comparison.json",
        {
            "schema_version": "qwen_cjt_q2_formal_comparison_v1",
            "protocol": fairness,
            "methods": summaries,
            "paired_nonzero": pairs,
            "bootstrap": {"seed": args.seed, "resamples": args.resamples, "unit": "scene-group"},
            "scope": (
                "paired synthetic pixel and template/shadow proxies; "
                "no semantic success or human rating"
            ),
        },
    )
    print({"output": str(output), "methods": list(summaries), "samples": fairness["samples"]})


def configure(parser) -> None:
    parser.add_argument("--method", action="append", required=True, help="LABEL=metrics.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.set_defaults(func=run)
