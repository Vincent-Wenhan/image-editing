"""A successful continuation must remain evaluable after the first attempt failed."""

import json

import pytest

from src.core.io import sha256
from src.qwen.provenance import completed_training_execution, verify_inference_execution


def receipt(path, status, returncode):
    path.write_text(json.dumps({"status": status, "returncode": returncode}))
    return path


def test_failed_initial_execution_accepts_successful_continuation(tmp_path):
    original = receipt(tmp_path / "execution.json", "failed", 1)
    continued = receipt(tmp_path / "execution_continue_120.json", "complete", 0)
    bound = completed_training_execution(tmp_path)
    assert bound == {"execution_receipt": str(continued), "execution_sha256": sha256(continued)}
    verify_inference_execution({"run": str(tmp_path), **bound})
    assert json.loads(original.read_text())["status"] == "failed"


def test_failed_continuation_keeps_completed_initial_execution_evaluable(tmp_path):
    original = receipt(tmp_path / "execution.json", "complete", 0)
    receipt(tmp_path / "execution_continue_240.json", "failed", 1)
    bound = completed_training_execution(tmp_path)
    assert bound["execution_receipt"] == str(original)


@pytest.mark.parametrize("status,returncode", [("failed", 1), ("running", None), ("complete", 1)])
def test_no_successful_execution_cannot_authorize_inference(tmp_path, status, returncode):
    receipt(tmp_path / "execution.json", status, returncode)
    receipt(tmp_path / "execution_continue_120.json", status, returncode)
    with pytest.raises(ValueError, match="completed training execution"):
        completed_training_execution(tmp_path)


def test_missing_execution_cannot_authorize_inference(tmp_path):
    with pytest.raises(ValueError, match="completed training execution"):
        completed_training_execution(tmp_path)


def test_changed_successful_receipt_is_rejected_after_inference(tmp_path):
    original = receipt(tmp_path / "execution.json", "complete", 0)
    inference = {"run": str(tmp_path), **completed_training_execution(tmp_path)}
    receipt(original, "failed", 1)
    with pytest.raises(ValueError, match="changed after inference"):
        verify_inference_execution(inference)


def test_another_runs_receipt_is_rejected(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    receipt(other / "execution.json", "complete", 0)
    inference = {"run": str(tmp_path), **completed_training_execution(other)}
    with pytest.raises(ValueError, match="another run"):
        verify_inference_execution(inference)


def test_binding_an_unsuccessful_receipt_is_rejected(tmp_path):
    path = receipt(tmp_path / "execution.json", "failed", 1)
    with pytest.raises(ValueError, match="unsuccessful"):
        verify_inference_execution(
            {"run": str(tmp_path), "execution_receipt": str(path), "execution_sha256": sha256(path)}
        )


@pytest.mark.parametrize("field", ["execution_receipt", "execution_sha256"])
def test_partial_receipt_binding_is_rejected(field):
    with pytest.raises(ValueError, match="Incomplete"):
        verify_inference_execution({field: "incomplete"})


def test_legacy_q2_inference_is_compatible():
    verify_inference_execution({"run": "legacy-q2"})
