"""Reject stale bytes or mismatched experiments before writing any evaluation results."""

import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.core.io import read_rows, sha256
from src.core.metadata import config_digest
from src.core.random import sample_seed
from src.evaluation.evaluate import run
from src.evaluation.provenance import verify_comparison


@pytest.fixture
def saved_run(paired_data, tmp_path):
    manifest = paired_data / "dev/manifest.csv"
    row = read_rows(manifest)[0]
    output = tmp_path / "predictions"
    output.mkdir()
    Image.new("RGB", (32, 48), "red").save(output / f"{row['id']}.png")
    config = dict(limit=1, side=48, steps=50, guidance=4.0, seed=42)
    record = dict(
        status="complete",
        config=config,
        config_sha256=config_digest(config),
        manifest_sha256=sha256(manifest),
        model=dict(repo_id="base", revision="fixed"),
        results=[
            dict(
                id=row["id"],
                width=32,
                height=48,
                seed=sample_seed(42, row["id"]),
                source_sha256=row["source_sha256"],
                prompt=row["instruction"],
            )
        ],
    )
    (output / "run.json").write_text(json.dumps(record))
    args = SimpleNamespace(
        data_root=paired_data,
        manifest=manifest,
        predictions=output,
        threshold=0.05,
        reference_run=None,
    )
    return args, record, row


@pytest.mark.parametrize("key", ["source", "target", "mask", "manifest"])
def test_changed_real_data_is_rejected_without_metrics(saved_run, key):
    args, record, row = saved_run
    if key == "manifest":
        args.manifest.write_text(args.manifest.read_text() + "\n")
    else:
        path = args.data_root / row[f"{key}_path"]
        Image.new("L" if key == "mask" else "RGB", (32, 48), 128).save(path)
    with pytest.raises(ValueError, match="SHA256|Checksum"):
        run(args)
    assert not (args.predictions / "metrics.json").exists()


@pytest.mark.parametrize("change", ["configuration", "prompt", "seed", "shape", "duplicate"])
def test_changed_experiment_identity_is_rejected(saved_run, change):
    args, record, row = saved_run
    if change == "configuration":
        record["config"]["guidance"] = 3.0
    elif change == "duplicate":
        record["results"].append(record["results"][0].copy())
    else:
        key = dict(prompt="prompt", seed="seed", shape="width")[change]
        record["results"][0][key] = "wrong" if key == "prompt" else 123
    (args.predictions / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        run(args)
    assert not (args.predictions / "metrics.json").exists()


def test_legacy_record_without_optional_config_digest_still_evaluates(saved_run):
    args, record, row = saved_run
    record.pop("config_sha256")
    (args.predictions / "run.json").write_text(json.dumps(record))
    run(args)
    assert json.loads((args.predictions / "metrics.json").read_text())["target_mae"] == 0


@pytest.mark.parametrize("key", ["steps", "guidance", "seed", "side"])
def test_paired_budget_mismatch_is_rejected(saved_run, key):
    args, reference, row = saved_run
    edited = copy.deepcopy(reference)
    edited["config"][key] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        verify_comparison(reference, edited)
