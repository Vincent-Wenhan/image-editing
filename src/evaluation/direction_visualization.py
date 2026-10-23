"""Non-blinded direction curves and all-scene contacts; never create human ratings."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.io import sha256, write_json, write_rows
from src.datasets.manifest import load_edit_input
from src.datasets.scene_manifest import load_role_supervision
from src.evaluation.direction_metrics import velocities
from src.evaluation.direction_provenance import arrays
from src.inference.direction_study import load_direction
from src.infra.experiment import source_snapshot


def saved(fig, folder, name):
    result = []
    for extension in ["png", "svg"]:
        path = folder / f"{name}.{extension}"
        fig.savefig(path, dpi=130)
        result.append(dict(path=str(path.resolve()), sha256=sha256(path)))
    return result


def curves(summary, folder, plt):
    output = []
    panels = [
        ("t_cosine", "T strength-direction cosine"),
        ("d_cosine", "D strength-direction cosine"),
        ("p_extra", "P endpoint MSE minus zero"),
        ("t_contrast_gain", "T strength contrast gain"),
        ("d_contrast_gain", "D strength contrast gain"),
        ("d_drop_d_gain", "Full D error gain / edit energy vs dropD"),
    ]
    groups = sorted({(r["family"], r["seed"], r["split"]) for r in summary["summaries"]})
    for family, seed, split in groups:
        selected = [
            r
            for r in summary["summaries"]
            if (r["family"], r["seed"], r["split"]) == (family, seed, split)
            and r["state_family"] == "source_bridge"
            and r["variant"] == "original"
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, (metric, title) in zip(axes.flat, panels, strict=True):
            methods = (
                ["full_role"]
                if metric == "d_drop_d_gain"
                else ["constant", "binary", "old_new", "full_role"]
            )
            for method in methods:
                for branch in (
                    ["cfg_native", "cfg_fp32"] if method == "full_role" else ["cfg_native"]
                ):
                    points = sorted(
                        [r for r in selected if r["method"] == method and r["branch"] == branch],
                        key=lambda r: r["time"],
                    )
                    values = [
                        r["metrics"]["p_endpoint_mse"] - r["metrics"]["p_zero_endpoint_mse"]
                        if metric == "p_extra"
                        else r["metrics"][metric]
                        for r in points
                    ]
                    values = [np.nan if v is None else v for v in values]
                    ax.plot(
                        [r["time"] for r in points],
                        values,
                        ("--" if branch == "cfg_fp32" else "-") + "o",
                        label=f"{method} {branch}",
                    )
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set(title=title, xlabel="Requested model time (actual BF16 values in records)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.suptitle(
            f"{family} / seed {seed} / {split}: target-free source bridge; "
            "latent proxies, no human scores"
        )
        output.extend(saved(fig, folder, f"{family}_{seed}_{split}"))
        plt.close(fig)
    return output


def contacts(summary, plan, folder, plt):
    from matplotlib.colors import TwoSlopeNorm

    output = []
    root = Path(plan["data"])
    settings = plan["inherited"]["inference"]
    for job in [j for j in plan["jobs"] if j["method"] == "full_role"]:
        receipt = summary["reports"][job["name"]]
        if sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Visualized report changed")
        report = json.loads(Path(receipt["path"]).read_text())
        old = json.loads(
            (Path(job["previous_job"]["directory"]) / "inference/report.json").read_text()
        )
        cases = {(r["id"], r["state"]["family"], r["state"]["time"]): r for r in report["results"]}
        for row in [r for r in plan["rows"] if float(r["strength"]) == 1]:
            group = report["groups"][row["group_id"]]
            neutral = next(
                r
                for r in plan["rows"]
                if r["group_id"] == row["group_id"] and float(r["strength"]) == 0
            )
            inputs = arrays(group["input_path"])
            target = arrays(group["supervision"][row["id"]]["path"])
            prior = next(
                r
                for r in old["results"]
                if r["id"] == row["id"]
                and r["update"] == job["pair"]["update"]
                and r["variant"] == "original"
            )
            if (
                sha256(prior["path"]) != prior["prediction_sha256"]
                or prior["checkpoint_sha256"] != job["pair"]["lora_sha256"]
            ):
                raise ValueError("Prior generated image/checkpoint changed")
            supervision = load_role_supervision(root, row)
            labels = np.asarray(supervision.role_labels)
            palette = np.array([[220, 70, 90], [250, 180, 40], [180, 205, 195]], np.uint8)
            region_image = np.zeros((*labels.shape, 3), np.uint8)
            for index in range(3):
                region_image[labels == index] = palette[index]
            with Image.open(prior["path"]) as image:
                prior_image = image.convert("RGB").copy()
            fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
            for ax, image, title in zip(
                axes[0],
                [load_edit_input(root, row).source, supervision.target, region_image, prior_image],
                [
                    "Source",
                    "Target (scoring only)",
                    "T/D/P scoring regions, not RGB input",
                    "Prior PR7 generation; no new generation",
                ],
                strict=True,
            ):
                ax.imshow(image)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            dot_maps = []
            gain_maps = []
            for time in plan["protocol"]["times"]:
                case = cases[(row["id"], "source_bridge", time)]
                identity = cases[(neutral["id"], "source_bridge", time)]
                actual = case["state"]["actual_time"]
                x = arrays(case["state"]["path"])["x"]
                full = velocities(
                    arrays(case["variants"]["original"]["path"]), settings["guidance"]
                )["cfg_native"]
                baseline = velocities(
                    arrays(identity["variants"]["original"]["path"]), settings["guidance"]
                )["cfg_native"]
                drop = velocities(arrays(case["variants"]["drop_d"]["path"]), settings["guidance"])[
                    "cfg_native"
                ]
                dot_maps.append(
                    ((actual * (full - baseline)) * (inputs["anchor"] - target["target"])).mean(1)
                )
                gain = (
                    (x - actual * drop - target["target"]) ** 2
                    - (x - actual * full - target["target"]) ** 2
                ).mean(1)
                gain_maps.append(
                    np.where(target["masks"][1] > 0, gain * target["masks"][1], np.nan)
                )
            height, width = int(row["height"]) // 16, int(row["width"]) // 16
            for index, (maps, label) in enumerate(
                [
                    (dot_maps, "Signed strength-direction dot; positive helps"),
                    (gain_maps, "D-weighted endpoint error gain vs dropD"),
                ],
                start=1,
            ):
                nonempty = [np.abs(a[np.isfinite(a)]) for a in maps if np.isfinite(a).any()]
                limit = max([float(a.max()) for a in nonempty] + [1e-12])
                for ax, a, time in zip(axes[index], maps, plan["protocol"]["times"], strict=True):
                    im = ax.imshow(
                        a.reshape(height, width),
                        cmap="coolwarm",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                        interpolation="nearest",
                    )
                    ax.set_title(f"{label}\nt={time:g}", fontsize=8)
                    ax.axis("off")
                fig.colorbar(
                    im,
                    ax=axes[index].tolist(),
                    shrink=0.6,
                    label="Actual latent units; shared scale across times",
                )
            fig.suptitle(
                f"{job['family']} / seed {job['seed']} / {row['id']}\n"
                "Non-blinded direction proxies; prior image is the preserved 20-step result",
                fontsize=10,
            )
            output.extend(saved(fig, folder, f"{job['family']}_{job['seed']}_{row['id']}"))
            plt.close(fig)
    return output


def automated_failures(summary, gate):
    output = []
    for row in summary["records"]:
        if (
            row["method"] != "full_role"
            or row["variant"] != "original"
            or row["branch"] != "cfg_native"
            or row["state_family"] != "source_bridge"
        ):
            continue
        labels = []
        if row["strength"] > 0:
            for region in ["t", "d"]:
                projection = row[f"{region}_projection"]
                if projection is None:
                    labels.append(f"{region}_direction_unavailable")
                elif projection < 0:
                    labels.append(f"{region}_reverse_direction")
                elif projection < gate["projection_min"]:
                    labels.append(f"{region}_weak_direction")
        if row["p_endpoint_mse"] - row["p_zero_endpoint_mse"] > max(
            gate["preservation_relative_max"] * row["p_zero_endpoint_mse"],
            gate["preservation_absolute_tolerance"],
        ):
            labels.append("p_extra_error")
        if labels:
            output.append(
                {
                    k: row[k]
                    for k in ["family", "seed", "split", "id", "group_id", "strength", "time"]
                }
                | dict(
                    categories=";".join(labels),
                    evaluation_type="automated_latent_direction_proxy_not_human_rating",
                )
            )
    return output


def export(study, analysis, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(output)
    plan = load_direction(study)
    summary = json.loads(analysis.read_text())
    if (
        summary["status"] != "complete"
        or summary["plan_sha256"] != sha256(study / "plan.json")
        or summary["test_status"] != "sealed"
    ):
        raise ValueError("Use a completed frozen analysis")
    output.mkdir(parents=True)
    for name in ["curves", "cases"]:
        (output / name).mkdir()
    files = curves(summary, output / "curves", plt) + contacts(summary, plan, output / "cases", plt)
    failures = automated_failures(summary, plan["protocol"]["gate"])
    if failures:
        write_rows(output / "automated_failures.csv", failures)
    cards = []
    for record in files:
        path = Path(record["path"])
        if path.suffix != ".png":
            continue
        relative = html.escape(str(path.relative_to(output.resolve())), quote=True)
        cards.append(
            f"<article><h2>{html.escape(path.stem)}</h2>"
            f'<img style="max-width:100%" src="{relative}"></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Frozen direction diagnostics</title>'
        "<p>Non-blinded latent proxies and preserved prior generations; "
        "no human scores, no new edited-image quality claim.</p>" + "".join(cards)
    )
    write_json(
        output / "index.json",
        dict(
            status="complete",
            source=source_snapshot(),
            summary_sha256=sha256(analysis),
            matplotlib=matplotlib.__version__,
            files=files,
            automatic_failure_rows=len(failures),
            human_status="pending",
            real_ratings=0,
            test_status="sealed",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.study, args.analysis, args.output)


if __name__ == "__main__":
    main()
