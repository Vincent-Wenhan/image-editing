"""One hash-bound encoding cache shared by matched Qwen loss variants."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.core.io import sha256, write_json
from src.core.metadata import tensor_sha
from src.qwen.role_data import PreparedSample, prepare_cache


def shared_cache(record, directory: Path, *, device, dtype) -> list[PreparedSample]:
    import torch

    manifest = Path(record["train_manifest"])
    root = Path(record["data_root"])
    policy = {
        "schema": "qwen_role_cache_v2_signed_shadow",
        "manifest_sha256": sha256(manifest),
        "dataset_sha256": sha256(root / "dataset.json"),
        "model": record["model"],
        "external_revision": record["external"]["revision"],
        "seed": record["seed"],
        "dtype": str(dtype),
        "implementation": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("role_cache.py", "role_data.py", "role_loss.py")
        },
    }
    tensor_path, receipt_path = directory / "cache.pt", directory / "cache.json"
    if directory.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["policy"] != policy or sha256(tensor_path) != receipt["sha256"]:
            raise ValueError("Shared Qwen cache policy or bytes changed; use a new cache directory")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        print(f"Reusing shared cache {directory}", flush=True)
        return [PreparedSample(**sample) for sample in payload]
    samples = prepare_cache(
        root, manifest, Path(record["model_path"]), device=device, dtype=dtype, seed=record["seed"]
    )
    directory.mkdir(parents=True)
    torch.save([asdict(sample) for sample in samples], tensor_path)
    write_json(
        receipt_path,
        {
            "policy": policy,
            "sha256": sha256(tensor_path),
            "samples": [
                {
                    "id": sample.sample_id,
                    **{
                        name: tensor_sha(value)
                        for name, value in asdict(sample).items()
                        if name != "sample_id"
                    },
                }
                for sample in samples
            ],
        },
    )
    return samples
