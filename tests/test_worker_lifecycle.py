"""Failures in training, telemetry and cleanup retain the original exception/report."""

import json
from types import SimpleNamespace

import pytest
import yaml

from src.core.config import ProjectSettings
from src.training import worker


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    root = tmp_path
    trainer = root / "external/ai-toolkit"
    trainer.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs/assets.lock.json").write_text(
        json.dumps({"repositories": {"ai-toolkit": {"revision": "fixed"}}})
    )
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(ProjectSettings.load().recipe))
    report = root / "report.json"
    monkeypatch.chdir(trainer)
    monkeypatch.setattr(worker, "require_lease", lambda: None)
    monkeypatch.setattr(worker, "command", lambda _: "fixed")
    monkeypatch.setattr(worker, "versions", lambda: {"test": "cpu"})
    monkeypatch.setattr(worker, "source_snapshot", lambda _: {"fixture": True})
    monkeypatch.setattr(worker, "validate_checkpoint", lambda _: {"checkpoint_finite": True})
    events = []

    class CUDA:
        OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        reset_peak_memory_stats = staticmethod(lambda: events.append("reset"))
        synchronize = staticmethod(lambda: events.append("sync"))
        max_memory_allocated = staticmethod(lambda: 8 * 2**30)
        max_memory_reserved = staticmethod(lambda: 10 * 2**30)
        get_device_properties = staticmethod(lambda _: SimpleNamespace(total_memory=48 * 2**30))
        mem_get_info = staticmethod(lambda: (38 * 2**30, 48 * 2**30))
        empty_cache = staticmethod(lambda: events.append("empty_cache"))

    class Job:
        process = [object()]

        def run(self):
            events.append("run")

        def cleanup(self):
            events.append("cleanup")

    job = Job()
    runtime = SimpleNamespace(
        torch=SimpleNamespace(cuda=CUDA), set_seed=lambda _: None, get_job=lambda _: job
    )
    return SimpleNamespace(
        root=root, config=config, report=report, job=job, runtime=runtime, events=events
    )


@pytest.mark.parametrize("failure", ["run", "construction", "checkpoint", "interrupt"])
def test_original_failure_survives_secondary_failures(worker_case, monkeypatch, failure):
    case = worker_case
    original = (
        KeyboardInterrupt("stop") if failure == "interrupt" else RuntimeError("original failure")
    )

    def fail(*args):
        raise original

    if failure == "construction":
        case.runtime.get_job = fail
    elif failure == "checkpoint":
        monkeypatch.setattr(worker, "validate_checkpoint", fail)
    else:
        case.job.run = fail

    def bad_cleanup():
        case.events.append("cleanup")
        raise RuntimeError("secondary cleanup failure")

    def bad_sync():
        raise RuntimeError("secondary CUDA synchronization failure")

    case.job.cleanup = bad_cleanup
    case.runtime.torch.cuda.synchronize = bad_sync
    with pytest.raises(type(original)) as captured:
        worker.run_training(
            case.config, case.report, case.root, runtime_loader=lambda *args: case.runtime
        )
    assert captured.value is original
    report = json.loads(case.report.read_text())
    assert report["status"] == "failed" and report["error"] == repr(original)
    assert "original failure" in report["traceback"] or failure == "interrupt"
    assert report["peak_reserved_gib"] == 10 and report["peak_allocated_gib"] == 8
    assert "synchronize" in report["telemetry_errors"]
    assert case.events[-1] == "empty_cache"
    if failure != "construction":
        assert "cleanup" in case.events and not case.job.process
        assert "job.cleanup" in report["cleanup_errors"]


def test_success_still_runs_cleanup_after_statistics(worker_case):
    case = worker_case
    worker.run_training(
        case.config, case.report, case.root, runtime_loader=lambda *args: case.runtime
    )
    report = json.loads(case.report.read_text())
    assert report["status"] == "complete" and report["checkpoint_finite"]
    assert report["gpu_total_gib"] == 48
    assert (
        case.events.index("sync") < case.events.index("cleanup") < case.events.index("empty_cache")
    )
    assert case.job.process == []


def test_cleanup_failure_is_reported_as_failure_after_success(worker_case):
    case = worker_case

    def fail():
        raise RuntimeError("cleanup failed")

    case.job.cleanup = fail
    with pytest.raises(RuntimeError, match="resource cleanup"):
        worker.run_training(
            case.config, case.report, case.root, runtime_loader=lambda *args: case.runtime
        )
    assert json.loads(case.report.read_text())["status"] == "failed"


def test_final_report_io_failure_does_not_mask_training_failure(worker_case, monkeypatch):
    case = worker_case
    original = RuntimeError("training failure")

    def fail():
        raise original

    def write(path, value):
        if value["status"] == "failed":
            raise OSError("disk full")
        path.write_text(json.dumps(value))

    case.job.run = fail
    monkeypatch.setattr(worker, "write_json", write)
    with pytest.raises(RuntimeError) as captured:
        worker.run_training(
            case.config, case.report, case.root, runtime_loader=lambda *args: case.runtime
        )
    assert captured.value is original
    assert case.events[-1] == "empty_cache"


def test_oom_is_explicit_and_cleanup_still_runs(worker_case):
    case = worker_case
    error = case.runtime.torch.cuda.OutOfMemoryError("not enough memory")

    def fail():
        raise error

    case.job.run = fail
    with pytest.raises(type(error)):
        worker.run_training(
            case.config, case.report, case.root, runtime_loader=lambda *args: case.runtime
        )
    assert json.loads(case.report.read_text())["out_of_memory"] is True
    assert "cleanup" in case.events and case.events[-1] == "empty_cache"


@pytest.mark.parametrize("close_fails", [False, True])
def test_feature_failure_keeps_exception_stats_and_releases_hooks(
    worker_case, monkeypatch, close_fails
):
    from src.training import feature_adapter, feature_config

    case = worker_case
    metadata = dict(mode="full_role")
    monkeypatch.setattr(feature_config, "verify_features", lambda _: metadata)
    original = RuntimeError("original feature training failure")

    def fail():
        raise original

    def close():
        case.events.append("feature_close")
        if close_fails:
            raise RuntimeError("feature hook cleanup failure")

    state = SimpleNamespace(
        stats=dict(batches=1, gradients={"fixture": {"finite": True}}), close=close
    )
    monkeypatch.setattr(feature_adapter, "install_features", lambda *args: state)
    case.runtime.model_type = type("ModelFixture", (), {})
    case.job.run = fail
    with pytest.raises(RuntimeError) as captured:
        worker.run_training(
            case.config, case.report, case.root, runtime_loader=lambda *args: case.runtime
        )
    assert captured.value is original
    report = json.loads(case.report.read_text())
    assert report["status"] == "failed" and report["role_features"] == metadata
    assert report["feature_statistics"] == state.stats
    assert report["peak_reserved_gib"] == 10
    assert case.events.index("feature_close") < case.events.index("cleanup")
    assert case.events[-1] == "empty_cache"
    assert bool(report.get("cleanup_errors")) == close_fails
