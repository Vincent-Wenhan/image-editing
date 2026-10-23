"""Failed attempts are immutable; reuse successful siblings without new GPU work."""

import json
from pathlib import Path

import pytest

from src.core.io import sha256, write_json
from src.inference import feature_diagnostic_study as study


def test_retry_retains_failed_attempt_and_reuses_completed_seed(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("frozen protocol fixture")
    p = dict(train_seeds=[211, 223], max_parallel=1)
    monkeypatch.setattr(study, "load_plan", lambda _path: dict(protocol=p))
    source = dict(revision="fixture", files={})
    calls = []

    def execute(argv, directory):
        seed = int(argv[argv.index("--train-seed") + 1])
        calls.append(seed)
        directory.mkdir()
        if seed == 223 and calls.count(seed) == 1:
            (directory / "failure.log").write_text("resource failure fixture, no GPU run")
            raise RuntimeError("resource unavailable fixture")
        output = Path(argv[argv.index("--output") + 1])
        write_json(
            output / "report.json",
            dict(
                status="complete",
                repeat_exact=True,
                source=source,
                plan_sha256=sha256(tmp_path / "plan.json"),
            ),
        )
        return dict(status="complete", returncode=0, source=source)

    monkeypatch.setattr(study, "execute", execute)
    # Write a valid smoke receipt; this is metadata fixture, not a real GPU claim.
    smoke = tmp_path / "smoke"
    write_json(
        smoke / "report.json",
        dict(
            status="complete",
            repeat_exact=True,
            source=source,
            plan_sha256=sha256(tmp_path / "plan.json"),
        ),
    )
    write_json(
        tmp_path / "smoke_phase.json",
        {
            "211": dict(
                output=str(smoke),
                report_sha256=sha256(smoke / "report.json"),
                execution=dict(status="complete", returncode=0, source=source),
            )
        },
    )
    with pytest.raises(RuntimeError, match="resource"):
        study.run_phase(tmp_path, "diagnose", 1)
    first = json.loads((tmp_path / "diagnose_211_receipt.json").read_text())
    study.run_phase(tmp_path, "diagnose", 2)
    assert calls.count(211) == 1 and calls.count(223) == 2
    assert json.loads((tmp_path / "diagnose_211_receipt.json").read_text()) == first
    assert (tmp_path / "diagnose_223_attempt01_execution/failure.log").exists()
    outputs = study.phase_outputs(tmp_path, "diagnose")
    assert "attempt02" in str(outputs["223"])
    with pytest.raises(FileExistsError):
        study.run_phase(tmp_path, "diagnose", 3)
    (outputs["223"] / "report.json").write_text("modified")
    with pytest.raises(ValueError, match="provenance"):
        study.phase_outputs(tmp_path, "diagnose")
