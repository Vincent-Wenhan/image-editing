"""Explicit oracle feature inference, independent of the image reference VAE."""

import json
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.backends.feature_fusion import feature_fusion
from src.backends.flux2 import Flux2Backend
from src.conditioning.features import RoleFeatures
from src.conditioning.static import validate_reference_size
from src.core.config import load_assets
from src.core.types import EditInput, EditOutput, GenerationOptions


class Flux2FeatureBackend:
    def __init__(self, model: str | Path, lora: str, encoder: str):
        self.encoder, self.metadata = load_encoder(Path(encoder), Path(lora))
        declared = json.loads((Path(model) / "download.json").read_text())
        expected = load_assets()["models"]["base"]
        if any(declared.get(key) != expected[key] for key in ("repo_id", "revision")):
            raise ValueError("Feature inference requires the pinned un-distilled Klein Base 4B")
        self.base = Flux2Backend(model, lora)
        self.encoder.to("cuda")
        self.encoder.eval()
        self.last_fusion = {}

    def edit_features(
        self,
        edit_input: EditInput,
        features: RoleFeatures,
        options: GenerationOptions,
        *,
        allow_oracle: bool = False,
    ) -> EditOutput:
        import torch

        if not allow_oracle:
            raise ValueError(
                "Independent role features require explicit oracle upper-bound consent"
            )
        if (
            features.mode != self.metadata["mode"]
            or edit_input.mask is not None
            or edit_input.strength is not None
            or features.size != edit_input.source.size
            or features.size != (options.width, options.height)
        ):
            raise ValueError("Feature mode/source/geometry mismatch or ambiguous extra conditions")
        validate_reference_size(features.size)
        tokens = torch.from_numpy(features.tokens())[None]
        grid = (options.height // 16, options.width // 16)
        with feature_fusion(self.base.pipe.transformer, self.encoder, tokens, grid) as stats:
            result = self.base._generate(edit_input, [edit_input.source], options, matched=True)
        if (
            stats["calls"] < options.steps
            or stats["source_tokens"] != [grid[0] * grid[1]]
            or stats["reference_times"] != [10]
        ):
            raise RuntimeError("Inference silently skipped role features")
        self.last_fusion = stats
        return result
