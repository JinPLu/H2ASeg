"""Serializable, fail-fast configuration contract for the MICCAI 2024 H2ASeg model.

This module deliberately has no PyTorch dependency.  It validates the tensor
shape assumptions made by ``model/`` before the comparatively expensive model
construction step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_SPATIAL_DIMS = 3
_ENCODER_STAGES = 5
_STRIDE_TWO_LEVELS = 4
_MCSA_CHANNEL_SCALE = 4
_MCSA_STAGES = (1, 2)  # zero-based encoder stages 2 and 3


def _positive_int_tuple(
    value: Sequence[int], *, length: int, name: str
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly {length} positive integers") from exc
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    return result


@dataclass(frozen=True)
class H2ASegModelConfig:
    """Model-construction values and the shape contract required by H2ASeg.

    The defaults reproduce the configuration used by the maintained MICCAI
    2024 training and inference entry points.  Only configurations compatible
    with the current model implementation are accepted.
    """

    patch_size: tuple[int, int, int] = (128, 128, 64)
    in_channels: int = 1
    n_classes: int = 2
    num_channels: tuple[int, int, int, int, int] = (8, 16, 32, 64, 128)
    window_size: tuple[int, int, int] = (8, 8, 4)
    attention_dropout: float = 0.5
    encoder_dropout: float = 0.5
    decoder_dropout: float = 0.5
    conv_bias: bool = True
    downsample_levels: int = _STRIDE_TWO_LEVELS

    def __post_init__(self) -> None:
        patch_size = _positive_int_tuple(
            self.patch_size, length=_SPATIAL_DIMS, name="patch_size"
        )
        window_size = _positive_int_tuple(
            self.window_size, length=_SPATIAL_DIMS, name="window_size"
        )
        num_channels = _positive_int_tuple(
            self.num_channels, length=_ENCODER_STAGES, name="num_channels"
        )
        object.__setattr__(self, "patch_size", patch_size)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "num_channels", num_channels)

        if self.in_channels != 1:
            raise ValueError(
                "in_channels must be 1 for the maintained single-channel CT/PET "
                "training and inference entry points"
            )
        # TAMW explicitly selects prediction channels 0 and 1 as background
        # and foreground.  Supporting other class counts would change its
        # semantics, so the current architecture is intentionally binary-only.
        if self.n_classes != 2:
            raise ValueError("n_classes must be 2 for the current binary H2ASeg architecture")
        if self.downsample_levels != _STRIDE_TWO_LEVELS:
            raise ValueError(
                f"downsample_levels must be {_STRIDE_TWO_LEVELS} for the current encoder"
            )
        if not isinstance(self.conv_bias, bool):
            raise ValueError("conv_bias must be a bool")
        for name in ("attention_dropout", "encoder_dropout", "decoder_dropout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a number in [0, 1]")

        patch_divisor = 2**self.downsample_levels
        if any(size % patch_divisor for size in patch_size):
            raise ValueError(
                "patch_size dimensions must be divisible by 16 so four stride-2 "
                "encoder levels and decoder skip connections remain aligned"
            )

        for stage in _MCSA_STAGES:
            feature_size = tuple(size // (2**stage) for size in patch_size)
            stage_number = stage + 1
            incompatible_axes = [
                axis
                for axis, (feature, window) in enumerate(zip(feature_size, window_size))
                if feature % window
            ]
            if incompatible_axes:
                raise ValueError(
                    f"MCSA stage {stage_number} feature size {feature_size} must be "
                    f"divisible axis-wise by window_size {window_size}"
                )

            pool_size = min(window_size)
            if any(feature % pool_size for feature in feature_size):
                raise ValueError(
                    f"MCSA stage {stage_number} feature size {feature_size} must be "
                    f"divisible on every axis by pool size min(window_size)={pool_size}"
                )

            if num_channels[stage] < _MCSA_CHANNEL_SCALE:
                raise ValueError(
                    f"num_channels[{stage}] must be at least {_MCSA_CHANNEL_SCALE}; "
                    f"MCSA reduces that stage by channel_scale={_MCSA_CHANNEL_SCALE}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe checkpoint metadata."""
        return {
            "patch_size": list(self.patch_size),
            "in_channels": self.in_channels,
            "n_classes": self.n_classes,
            "num_channels": list(self.num_channels),
            "window_size": list(self.window_size),
            "attention_dropout": self.attention_dropout,
            "encoder_dropout": self.encoder_dropout,
            "decoder_dropout": self.decoder_dropout,
            "conv_bias": self.conv_bias,
            "downsample_levels": self.downsample_levels,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "H2ASegModelConfig":
        """Validate and reconstruct configuration stored in checkpoint metadata."""
        if not isinstance(values, Mapping):
            raise TypeError("model configuration must be a mapping")
        return cls(**dict(values))

    def to_model_kwargs(self) -> dict[str, Any]:
        """Return only arguments accepted by ``model.H2ASeg.H2ASeg``."""
        values = self.to_dict()
        values.pop("downsample_levels")
        return values


DEFAULT_MODEL_CONFIG = H2ASegModelConfig()
