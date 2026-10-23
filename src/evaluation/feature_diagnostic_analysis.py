"""SHA-bound region, position and causal-shadow diagnostics for frozen feature studies."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.core.io import read_rows, sha256, write_json, write_rows
from src.datasets.scene_manifest import load_role_supervision
from src.datasets.scene_pairs import rgb_image
from src.evaluation.feature_analysis import scene_mean
from src.evaluation.provenance import verify_run
from src.evaluation.role_failures import geometry_metrics, pixels
from src.evaluation.role_metrics import aggregate, region_metrics
from src.evaluation.scene_semantics import TranslationLocator, change_proxies
from src.inference.feature_diagnostic_study import load_plan, phase_outputs
from src.infra.experiment import source_snapshot

EXTRA_METRICS = (
    "old_only_target_mae",
    "new_only_target_mae",
    "position_error_px",
    "position_error_extents",
    "d_shadow_removal_mae",
    "d_shadow_creation_mae",
    "d_signed_change_response",
    "d_dark_region_iou",
    "p_edge_mae",
)


def scene_references(root: Path, split: str = "dev", identifiers: set[str] | None = None) -> dict:
    """Evaluation-only references; target/background never enter the inference worker."""
    if split not in {"train", "dev"}:
        raise ValueError("Research semantic diagnostics cannot inspect sealed Test")
    result = {}
    for row in read_rows(root / split / "manifest.csv"):
        if identifiers is not None and row["id"] not in identifiers:
            continue
        supervision = load_role_supervision(root, row)
        size = int(row["width"]), int(row["height"])
        source, target = [pixels(root / row[k], size) for k in ["source_path", "target_path"]]
        evidence = root / row["evidence_path"]
        if sha256(evidence) != row["evidence_sha256"]:
            raise ValueError("Semantic reference evidence SHA256 changed")
        with np.load(evidence, allow_pickle=False) as arrays:
            old, new = [arrays[k] > 0.5 for k in ["source_occupancy", "target_occupancy"]]
            background = np.asarray(rgb_image(arrays["background"]), dtype=np.float32) / 255
        labels, valid = (
            np.asarray(supervision.role_labels),
            np.asarray(supervision.valid_region) > 0,
        )
        result[row["id"]] = dict(
            row=row,
            source=source,
            target=target,
            background=background,
            old=old,
            new=new,
            labels=labels,
            valid=valid,
            locator=TranslationLocator(source, target, background, old, new),
        )
    if identifiers is not None and set(result) != identifiers:
        raise ValueError("Requested semantic examples differ from the selected split")
    return result


def score_prediction(reference: dict, prediction: np.ndarray) -> dict:
    r = reference
    located = r["locator"].locate(prediction)
    return dict(
        **region_metrics(r["source"], r["target"], prediction, r["labels"], r["valid"]),
        **geometry_metrics(
            r["source"],
            r["target"],
            prediction,
            dict(old=r["old"] & r["valid"], new=r["new"] & r["valid"]),
        ),
        **change_proxies(
            r["source"], r["target"], prediction, r["background"], r["labels"], r["valid"]
        ),
        position_error_px=located["position_error_px"],
        position_error_extents=located["position_error_extents"],
        localization=located,
    )


def summarize_records(rows: list[dict]) -> dict:
    def summarize(selected):
        return dict(
            **aggregate(selected),
            **{key: scene_mean(selected, key) for key in EXTRA_METRICS},
            localized=sum(row["localization"]["status"] == "localized" for row in selected),
            abstained=sum(row["localization"]["status"] == "abstain" for row in selected),
            duplicate_candidates=[
                row["id"]
                for row in selected
                if row["localization"]["template_duplicate_candidate"] is True
            ],
        )

    return dict(
        records=rows,
        all=summarize(rows),
        nonzero=summarize([r for r in rows if r["strength"] > 0]),
        by_strength={
            str(s): summarize([r for r in rows if r["strength"] == s])
            for s in sorted({r["strength"] for r in rows})
        },
    )


def analyze(study: Path, output: Path, phase: str, original_analysis: Path | None = None):
    if output.exists():
        raise FileExistsError(output)
    plan = load_plan(study)
    if phase == "existing" and original_analysis is None:
        raise ValueError("Bind existing predictions to the original hash-verified analysis")
    original = json.loads(original_analysis.read_text()) if original_analysis else None
    root = Path(plan["data"])
    references = scene_references(root)
    result = dict(
        status="running",
        phase=phase,
        plan_sha256=sha256(study / "plan.json"),
        data=str(root.resolve()),
        manifest_sha256=sha256(root / "dev/manifest.csv"),
        human_status="pending",
        real_ratings=0,
        scope="Synthetic pixel/template proxies; candidates are not object counts or human scores",
        source=source_snapshot(),
        calibration={k: r["locator"].calibration for k, r in references.items()},
        seeds={},
    )
    output.mkdir(parents=True)
    for seed in plan["protocol"]["train_seeds"]:
        methods = {}
        checkpoints = [c for c in plan["checkpoints"] if c["seed"] == seed]
        if phase == "existing":
            entries = []
            for checkpoint in checkpoints:
                directory = Path(checkpoint["previous_predictions"])
                record = json.loads((directory / "run.json").read_text())
                verify_run(root, root / "dev/manifest.csv", record)
                audit = original["seeds"][str(seed)]["fairness"]["methods"][checkpoint["method"]]
                scored = original["seeds"][str(seed)]["methods"][checkpoint["method"]]
                budget = plan["protocol"]["inference"]
                if (
                    record["lora_sha256"] != checkpoint["lora_sha256"]
                    or audit["lora_sha256"] != checkpoint["lora_sha256"]
                    or audit["encoder_sha256"] != checkpoint["encoder_sha256"]
                    or scored["run_sha256"] != sha256(directory / "run.json")
                    or any(record["config"][k] != budget[k] for k in ["steps", "guidance", "seed"])
                    or record["config"]["side"] != budget["side"]
                ):
                    raise ValueError("Semantic baseline differs from original matched checkpoint")
                for item in record["results"]:
                    if (
                        sha256(directory / f"{item['id']}.png")
                        != scored["prediction_sha256"][item["id"]]
                    ):
                        raise ValueError(
                            "Original prediction bytes changed since the verified study"
                        )
                entries += [
                    dict(
                        item,
                        variant=checkpoint["method"],
                        path=directory / f"{item['id']}.png",
                        run_sha256=sha256(directory / "run.json"),
                    )
                    for item in record["results"]
                ]
        else:
            directory = phase_outputs(study, phase)[str(seed)]
            report = json.loads((directory / "report.json").read_text())
            checkpoint = next(c for c in checkpoints if c["method"] == "full_role")
            if (
                report["status"] != "complete"
                or report["checkpoint"] != checkpoint
                or report["plan_sha256"] != sha256(study / "plan.json")
                or report["training_updates"] != 0
            ):
                raise ValueError("Incomplete or mismatched intervention experiment")
            expected_ids = (
                set(references)
                if phase == "compare"
                else set(plan["protocol"]["diagnostic_samples"])
            )
            if phase == "compare":
                from src.inference.feature_repair_study import load_comparison

                _, comparison = load_comparison(study)
                variants = comparison["recipe"]["variants"]
                if report["comparison_sha256"] != sha256(study / "comparison.json"):
                    raise ValueError("Repair report differs from its preregistered recipe")
                for name in variants:
                    run = json.loads((directory / name / "run.json").read_text())
                    verify_run(root, root / "dev/manifest.csv", run)
                    selected = [r for r in report["results"] if r["variant"] == name]
                    if (
                        run["results"] != selected
                        or run["residual_intervention"] != name
                        or run["lora_sha256"] != checkpoint["lora_sha256"]
                        or run["role_features"]["sha256"] != checkpoint["encoder_sha256"]
                        or any(
                            run["config"][k] != plan["protocol"]["inference"][k]
                            for k in ("side", "steps", "guidance", "seed")
                        )
                    ):
                        raise ValueError("Repair run differs from its actual checkpoint/budget")
            else:
                variants = plan["protocol"]["variants"]
            declared = [(r["id"], r["variant"]) for r in report["results"]]
            expected = {(identifier, name) for identifier in expected_ids for name in variants}
            if set(declared) != expected or len(declared) != len(expected):
                raise ValueError("Incomplete or duplicate intervention matrix")
            entries = [
                dict(
                    item,
                    path=directory
                    / item.get("prediction_path", f"{item['id']}/{item['variant']}/prediction.png"),
                    run_sha256=sha256(directory / "report.json"),
                )
                for item in report["results"]
            ]
        for entry in entries:
            r, path = references[entry["id"]], Path(entry["path"])
            if entry["source_sha256"] != r["row"]["source_sha256"] or (
                entry.get("prediction_sha256") and sha256(path) != entry["prediction_sha256"]
            ):
                raise ValueError("Actual diagnostic prediction/source SHA256 changed")
            prediction = pixels(path, (int(r["row"]["width"]), int(r["row"]["height"])))
            record = dict(
                id=entry["id"],
                group_id=r["row"]["group_id"],
                strength=float(r["row"]["strength"]),
                variant=entry["variant"],
                prediction_path=str(path.resolve()),
                prediction_sha256=sha256(path),
                run_sha256=entry["run_sha256"],
                **score_prediction(r, prediction),
            )
            methods.setdefault(entry["variant"], []).append(record)
        result["seeds"][str(seed)] = {m: summarize_records(rows) for m, rows in methods.items()}
        for name, rows in methods.items():
            write_rows(
                output / f"{seed}_{name}.csv",
                [dict(r, localization=json.dumps(r["localization"])) for r in rows],
            )
    result["status"] = "complete"
    if original_analysis:
        result["original_analysis_sha256"] = sha256(original_analysis)
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["existing", "diagnose", "compare"], required=True)
    parser.add_argument("--original-analysis", type=Path)
    args = parser.parse_args()
    analyze(args.study, args.output, args.phase, args.original_analysis)


if __name__ == "__main__":
    main()
