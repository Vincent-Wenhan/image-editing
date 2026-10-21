"""Asset waiting failures retain the existing status.json contract and start no jobs."""

import json
import subprocess
import sys

import pytest
import yaml

from src.infra import acceptance


def test_acceptance_timeout_records_failure_without_starting_gpu(tmp_path, monkeypatch):
    output = tmp_path / "acceptance"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/assets.lock.json").write_text(
        json.dumps({"models": {"base": {"path": "missing"}}, "dataset": {"revision": "test"}})
    )
    monkeypatch.setattr(acceptance, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["run_stage0.py", "--wait-hours", "0", "--output", str(output)]
    )
    calls = []
    monkeypatch.setattr(acceptance.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(TimeoutError, match="incomplete"):
        acceptance.main()

    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "failed" and state["completed"] == []
    assert set(state) == {
        "created",
        "status",
        "completed",
        "data_root",
        "error",
        "updated",
        "seconds",
    }
    assert calls == []
    with pytest.raises(FileExistsError):
        acceptance.main()


@pytest.mark.parametrize("oom", [True, False])
def test_only_grown_batch_oom_retries_in_new_experiment(paired_data, tmp_path, monkeypatch, oom):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    model = root / "models/base"
    model.mkdir(parents=True)
    lock = dict(
        models={"base": dict(path="models/base", revision="r")}, dataset=dict(revision="dataset")
    )
    (root / "configs/assets.lock.json").write_text(json.dumps(lock))
    (model / "download.json").write_text('{"revision":"r"}')
    (paired_data / "dataset.json").write_text('{"revision":"dataset","status":"complete"}')
    output = tmp_path / "acceptance"
    monkeypatch.setattr(acceptance, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage0.py",
            "--data-root",
            str(paired_data),
            "--output",
            str(output),
            "--wait-hours",
            "0",
        ],
    )
    attempts, checkpoints = [], []

    def execute(argv, **kwargs):
        if "train-config" in argv:
            name = argv[argv.index("--name") + 1]
            batch = int(argv[argv.index("--batch-size") + 1])
            path = output / f"{name}.yaml"
            path.write_text(yaml.safe_dump(dict(name=name, batch=batch)))
        elif "src.training" in argv:
            config = yaml.safe_load((output / argv[argv.index("src.training") + 1]).read_text())
            name = config["name"]
            report = output / f"{name}_report.json"
            attempts.append(name)
            if name == "smoke":
                report.write_text(
                    json.dumps(dict(status="failed", out_of_memory=oom, batch_size=4))
                )
                raise subprocess.CalledProcessError(1, argv)
            checkpoint = output / f"{name}.safetensors"
            checkpoint.touch()
            report.write_text(json.dumps(dict(status="complete", checkpoint=str(checkpoint))))
        elif "infer" in argv and "--lora" in argv:
            checkpoints.append(argv[argv.index("--lora") + 1])

    monkeypatch.setattr(acceptance.subprocess, "run", execute)
    if oom:
        acceptance.main()
        assert attempts == ["debug", "smoke", "smoke_retry"]
        assert checkpoints == [str(output / "smoke_retry.safetensors")]
        assert json.loads((output / "smoke_report.json").read_text())["status"] == "failed"
        assert json.loads((output / "status.json").read_text())["status"] == "complete"
    else:
        with pytest.raises(subprocess.CalledProcessError):
            acceptance.main()
        assert attempts == ["debug", "smoke"] and checkpoints == []
