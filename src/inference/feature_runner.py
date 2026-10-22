"""Dev-only, hash-bound feature inference; saved records extend the existing format."""

import argparse
import json
import traceback
from pathlib import Path

from src.backends.flux2_features import Flux2FeatureBackend
from src.conditioning.features import FEATURE_MODES
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import config_digest, config_snapshot, now, versions
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.features import load_features
from src.datasets.holdout import require_dev
from src.datasets.manifest import load_edit_input
from src.infra.gpu import require_lease


def feature_identity(features) -> dict:
    return dict(features.metadata, feature_sha256=features.digest, mode=features.mode)


def run(args, *, backend_factory=Flux2FeatureBackend):
    require_lease()
    root, output, model = Path(args.data_root), Path(args.output), Path(args.model)
    require_dev(root, Path(args.cohort), Path(args.manifest))
    if not args.allow_oracle or min(args.limit, args.steps) < 1:
        raise ValueError("Positive limits and explicit --allow-oracle required")
    rows = read_rows(args.manifest)[: args.limit]
    if any((int(r["width"]), int(r["height"])) != (args.side, args.side) for r in rows):
        raise ValueError("Feature inference cannot resize scene truth")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = dict(
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config_snapshot(args).items()
        },
        control="none",
        reference_preprocessing="training",
    )
    metadata = dict(
        path=str(Path(args.encoder).resolve()),
        sha256=sha256(args.encoder),
        mode=args.feature_mode,
        cohort=str(Path(args.cohort).resolve()),
        cohort_sha256=sha256(args.cohort),
        image_order="source",
        feature_tokens=(args.side // 16) ** 2,
    )
    record = dict(
        created=now(),
        status="running",
        config=config,
        config_sha256=config_digest(config),
        model=json.loads((model / "download.json").read_text()),
        packages=versions(),
        manifest_sha256=sha256(args.manifest),
        lora_sha256=sha256(args.lora),
        role_features=metadata,
        reference_preprocessing="training",
        results=[],
    )
    write_json(output / "run.json", record)
    try:
        backend = backend_factory(model, args.lora, args.encoder)
        if backend.metadata["mode"] != args.feature_mode:
            raise ValueError("Feature checkpoint mode differs from inference configuration")
        metadata["checkpoint_metadata"] = backend.metadata
        for row in rows:
            features = load_features(root, row, args.feature_mode)
            options = GenerationOptions(
                args.side, args.side, args.steps, args.guidance, sample_seed(args.seed, row["id"])
            )
            prediction = backend.edit_features(
                load_edit_input(root, row), features, options, allow_oracle=True
            )
            if prediction.image.size != (args.side, args.side):
                raise ValueError("Feature prediction size changed")
            prediction.image.save(output / f"{row['id']}.png")
            record["results"].append(
                dict(
                    id=row["id"],
                    seed=options.seed,
                    width=options.width,
                    height=options.height,
                    source_sha256=row["source_sha256"],
                    prompt=row["instruction"],
                    seconds=prediction.seconds,
                    peak_allocated_gib=prediction.peak_allocated_gib,
                    peak_reserved_gib=prediction.peak_reserved_gib,
                    role_features=feature_identity(features),
                    fusion=backend.last_fusion,
                )
            )
            write_json(output / "run.json", record)
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "run.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model", default=ProjectSettings.load().assets["models"]["base"]["path"])
    parser.add_argument("--lora", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, required=True)
    parser.add_argument("--allow-oracle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
