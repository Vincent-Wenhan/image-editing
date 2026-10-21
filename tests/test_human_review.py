"""Blinding, incomplete judgements and image integrity for human assessment."""

import json
from pathlib import Path

import pytest

from src.cli import build_parser
from src.core.io import read_rows, write_rows
from src.core.types import EditOutput
from src.evaluation.review import export_review, score_review
from src.inference import runner


@pytest.fixture
def review_bundle(spatial_data, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "require_lease", lambda: None)
    monkeypatch.setattr(runner, "versions", lambda: {"test": "cpu"})
    model = tmp_path / "model"
    model.mkdir()
    (model / "download.json").write_text('{"revision":"fixed"}')

    class Backend:
        def __init__(self, *args):
            pass

        def edit(self, edit, controls, options):
            return EditOutput(edit.source, 0, 0, 0)

    runs = {}
    for name in ("private_method_alpha", "private_method_beta"):
        runs[name] = tmp_path / name
        args = build_parser().parse_args(
            [
                "infer",
                "--data-root",
                str(spatial_data),
                "--manifest",
                str(spatial_data / "dev/manifest.csv"),
                "--model",
                str(model),
                "--side",
                "64",
                "--output",
                str(runs[name]),
                "--limit",
                "3",
            ]
        )
        runner.run(args, backend_factory=Backend)
    bundle = tmp_path / "review"
    export_review(spatial_data, spatial_data / "dev/manifest.csv", runs, bundle, 42)
    return bundle


def test_review_is_blinded_pending_and_roundtrips_bom(review_bundle, tmp_path):
    page = (review_bundle / "index.html").read_text()
    assert "private_method_alpha" not in page and "private_method_beta" not in page
    assert "\\n" in page  # JavaScript escapes must survive Python string generation.
    assert len(list((review_bundle / "images").iterdir())) == 18
    output = tmp_path / "pending.json"
    pending = score_review(review_bundle, review_bundle / "ratings.csv", output)
    assert pending["status"] == "pending" and pending["pending"] == 6
    assert all(info["edit_success"]["mean"] is None for info in pending["methods"].values())
    ratings = read_rows(review_bundle / "ratings.csv")
    for row in ratings:
        row.update(
            edit_success="2",
            preservation="1",
            reviewer="test-human",
            failure="collateral_change",
            comment="test fixture judgement",
        )
    path = tmp_path / "ratings.csv"
    write_rows(path, ratings)
    path.write_text("\ufeff" + path.read_text())
    result = score_review(review_bundle, path, tmp_path / "scored.json")
    assert result["status"] == "complete" and len(result["failure_cases"]) == 6
    assert all(info["preservation"]["mean"] == 1 for info in result["methods"].values())
    with pytest.raises(FileExistsError):
        score_review(review_bundle, path, output)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "score", "partial", "reviewer", "image"]
)
def test_invalid_or_tampered_human_ratings_fail(review_bundle, tmp_path, change):
    rows = read_rows(review_bundle / "ratings.csv")
    for row in rows:
        row.update(edit_success="2", preservation="2", reviewer="human")
    if change == "duplicate":
        rows[1] = rows[0].copy()
    elif change == "missing":
        rows.pop()
    elif change == "score":
        rows[0]["edit_success"] = "3"
    elif change == "partial":
        rows[0]["preservation"] = ""
    elif change == "reviewer":
        rows[0]["reviewer"] = ""
    else:
        mapping = json.loads((review_bundle / "private/mapping.json").read_text())
        Path(mapping["tasks"][0]["files"]["prediction"]["path"]).write_bytes(b"changed")
    path = tmp_path / "ratings.csv"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        score_review(review_bundle, path, tmp_path / "bad.json")
