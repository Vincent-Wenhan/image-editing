"""Frozen plans reject changed prompts, missing comparisons, Test rows and changed weights."""

import copy

import pytest

from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.inference import direction_study


@pytest.fixture
def frozen_plan(tmp_path, monkeypatch):
    inherited = load_yaml(CONFIG_ROOT / "experiments/role_learnability.yaml")
    protocol = load_yaml(CONFIG_ROOT / "experiments/role_direction.yaml")
    data = tmp_path / "data"
    for split in ["train", "dev", "test"]:
        (data / split).mkdir(parents=True)
        (data / split / "manifest.csv").write_text("id,split\n")
    rows = [dict(id="train_1", group_id="scene1", split="train", strength="0.5")]
    parents, jobs = {}, []
    for family in ["original", "channel_rms"]:
        folder = tmp_path / family
        folder.mkdir()
        old_jobs = []
        for method in inherited["methods"]:
            for seed in inherited["train_seeds"]:
                directory = folder / f"{method}_{seed}"
                directory.mkdir()
                lora = directory / "model.safetensors"
                lora.write_bytes(b"fixed existing checkpoint")
                encoder = directory / "features.safetensors"
                encoder.write_bytes(b"fixed existing encoder")
                pair = dict(
                    update=120,
                    lora=str(lora),
                    lora_sha256=sha256(lora),
                    encoder=dict(path=str(encoder), sha256=sha256(encoder)),
                )
                report = dict(
                    status="complete", feature_statistics=dict(observation=dict(checkpoints=[pair]))
                )
                write_json(directory / "train_report.json", report)
                old = dict(
                    name=f"train_{method}_{seed}",
                    phase="train",
                    method=method,
                    seed=seed,
                    directory=str(directory),
                )
                old_jobs.append(old)
                jobs.append(
                    dict(
                        name=f"{family}_{method}_{seed}",
                        family=family,
                        method=method,
                        seed=seed,
                        previous_job=old,
                        pair=pair,
                        train_report_sha256=sha256(directory / "train_report.json"),
                    )
                )
        write_json(folder / "plan.json", dict(jobs=old_jobs, protocol=inherited, data=str(data)))
        parents[family] = dict(path=str(folder), sha256=sha256(folder / "plan.json"))
    plan = dict(
        status="ready",
        training_updates=0,
        test_status="sealed",
        protocol=protocol,
        inherited=inherited,
        data=str(data),
        rows=rows,
        parents=parents,
        jobs=jobs,
        manifests={s: sha256(data / s / "manifest.csv") for s in ["train", "dev", "test"]},
    )
    monkeypatch.setattr(direction_study, "evaluation_rows", lambda *args: copy.deepcopy(rows))
    return tmp_path, plan


def test_frozen_binding_accepts_only_existing_complete_matrix(frozen_plan):
    folder, plan = frozen_plan
    write_json(folder / "plan.json", plan)
    assert len(direction_study.load_direction(folder)["jobs"]) == 16
    for job in plan["jobs"]:
        direction_study.verify_job(job)
    specs = direction_study.state_specs(plan["protocol"])
    assert (
        len(specs) == 7
        and specs.count(("source_bridge", 1.0)) == 1
        and ("noise_only", 1.0) not in specs
    )


@pytest.mark.parametrize(
    "change", ["prompt", "missing", "duplicate", "test", "manifest", "checkpoint", "pair"]
)
def test_frozen_binding_rejects_tampering(frozen_plan, change):
    from pathlib import Path

    folder, original = frozen_plan
    plan = copy.deepcopy(original)
    if change == "prompt":
        plan["inherited"]["prompt"] = "a different target hint"
    elif change == "missing":
        plan["jobs"].pop()
    elif change == "duplicate":
        plan["jobs"][0] = copy.deepcopy(plan["jobs"][1])
    elif change == "test":
        plan["rows"].append(dict(id="sealed", split="test"))
    elif change == "manifest":
        (Path(plan["data"]) / "train/manifest.csv").write_text("changed")
    elif change == "checkpoint":
        Path(plan["jobs"][0]["pair"]["lora"]).write_bytes(b"modified weights")
    elif change == "pair":
        plan["jobs"][0]["pair"]["update"] = 20
    write_json(folder / "plan.json", plan)
    with pytest.raises(ValueError):
        actual = direction_study.load_direction(folder)
        for job in actual["jobs"]:
            direction_study.verify_job(job)
