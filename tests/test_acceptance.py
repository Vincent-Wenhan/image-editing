"""Asset waiting failures retain the existing status.json contract and start no jobs."""

import json
import sys

import pytest

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
    assert set(state) == {"created", "status", "completed", "data_root", "error", "updated"}
    assert calls == []
    with pytest.raises(FileExistsError):
        acceptance.main()
