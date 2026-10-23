"""Non-RGB, explicit oracle fields. Channel order is old/new/D/P/valid."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

import numpy as np

FEATURE_MODES = ("constant", "binary", "old_new", "full_role")
FEATURE_SCHEMA = "aligned_noise_residual_v1"


@dataclass(frozen=True)
class EncoderConfig:
    hidden: int = 64
    channels: int = 5
    output: int = 128
    stride: int = 16
    max_residual: float = 0.25

    def __post_init__(self):
        if (
            type(self.hidden) is not int
            or not 1 <= self.hidden <= 512
            or any(type(value) is not int for value in (self.channels, self.output, self.stride))
            or (self.channels, self.output, self.stride) != (5, 128, 16)
            or isinstance(self.max_residual, bool)
            or not np.isfinite(self.max_residual)
            or not 0 < self.max_residual <= 1
        ):
            raise ValueError("Unsupported FLUX role feature configuration")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoleFeatures:
    values: np.ndarray
    mode: str = "full_role"
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        a = np.asarray(self.values)
        if (
            self.mode not in FEATURE_MODES
            or a.ndim != 3
            or a.shape[0] != 5
            or min(a.shape[1:]) < 64
            or any(n % 16 for n in a.shape[1:])
            or a.dtype.kind != "f"
            or not np.isfinite(a).all()
            or a.min() < 0
            or a.max() > 1
            or (a[:4, a[4] == 0] != 0).any()
            or (a[:4] > a[4:5] + 1e-6).any()
            or (self.mode == "full_role" and (a[2] + a[3] > a[4] + 1e-6).any())
        ):
            raise ValueError("Features require finite aligned 5xHxW unit maps, masked by valid")
        if self.metadata.get("purpose") != "oracle_upper_bound":
            raise ValueError("Role features require explicit oracle upper-bound provenance")
        if any(
            v is not None and not isinstance(v, str | int | float | bool)
            for v in self.metadata.values()
        ):
            raise TypeError("Feature metadata must be scalar, never target images")
        frozen = np.array(a, dtype=np.float32, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def size(self) -> tuple[int, int]:
        return self.values.shape[2], self.values.shape[1]

    def tokens(self) -> np.ndarray:
        _, height, width = self.values.shape
        return (
            self.values.reshape(5, height // 16, 16, width // 16, 16)
            .mean((2, 4))
            .transpose(1, 2, 0)
            .reshape(-1, 5)
            .copy()
        )

    @property
    def digest(self) -> str:
        header = json.dumps(
            dict(mode=self.mode, shape=self.values.shape, schema=FEATURE_SCHEMA), sort_keys=True
        )
        return hashlib.sha256(header.encode() + self.values.tobytes()).hexdigest()
