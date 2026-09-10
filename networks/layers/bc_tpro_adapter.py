"""Raw-frame background-conditioned temporal evidence for DeepPro-Plus.

Stage 1 intentionally uses a static evidence projection: there is no
input-dependent router or gate.  The variants differ only in the evidence
given to a compact ``3/9 -> 8 -> 32`` residual branch.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_VARIANTS = (
    "none",
    "temporal_control",
    "center_multiscale",
    "center_ring",
    "center_ring_difference",
    "temporal_bandpass",
    "center_spatial_smooth",
)


def _validate_raw_frames(raw_frames: torch.Tensor) -> None:
    if raw_frames.ndim != 5 or raw_frames.shape[1] != 1:
        raise ValueError(
            "raw_frames must have shape [B, 1, T, H, W], got %s"
            % (tuple(raw_frames.shape),)
        )


def _valid_frame_mask(raw_frames: torch.Tensor) -> torch.Tensor:
    """Identify exact all-zero temporal padding inserted by data loaders."""
    _validate_raw_frames(raw_frames)
    return raw_frames.detach().abs().amax(
        dim=(1, 3, 4), keepdim=True
    ).ne(0)


def _masked_temporal_statistics(
    signal: torch.Tensor,
    valid_mask: torch.Tensor,
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and count over valid samples in an odd centred window.

    The centre sample is included.  Temporal array boundaries and all-zero
    loader padding do not enter either the numerator or denominator.
    """
    if signal.ndim != 5:
        raise ValueError("signal must have shape [B, C, T, H, W]")
    expected_mask_shape = (signal.shape[0], 1, signal.shape[2], 1, 1)
    if valid_mask.shape != expected_mask_shape:
        raise ValueError(
            "valid_mask has shape %s, expected %s"
            % (tuple(valid_mask.shape), expected_mask_shape)
        )
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than one")

    valid = valid_mask.to(dtype=signal.dtype)
    radius = kernel_size // 2
    pooling_kernel = (kernel_size, 1, 1)
    # avg_pool is used for both terms, so its 1/kernel_size scale cancels.
    numerator = F.avg_pool3d(
        F.pad(signal * valid, (0, 0, 0, 0, radius, radius)),
        kernel_size=pooling_kernel,
        stride=1,
    )
    count_fraction = F.avg_pool3d(
        F.pad(valid, (0, 0, 0, 0, radius, radius)),
        kernel_size=pooling_kernel,
        stride=1,
    )
    mean = torch.where(
        count_fraction > 0,
        numerator / count_fraction.clamp_min(
            torch.finfo(signal.dtype).eps
        ),
        torch.zeros_like(numerator),
    )
    valid_count = count_fraction * float(kernel_size)
    return mean, valid_count


def _unit_l2_zero_dc_response(
    signal: torch.Tensor,
    valid_mask: torch.Tensor,
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply a boundary-normalized centre-minus-window-mean kernel.

    For ``k'`` valid samples, the multiplier ``sqrt(k'/(k'-1))`` gives
    ``delta_center - uniform_mean`` unit L2 norm while retaining zero DC.  A
    query with fewer than two valid samples has no temporal evidence and is
    explicitly zeroed.
    """
    mean, valid_count = _masked_temporal_statistics(
        signal, valid_mask, kernel_size
    )
    usable = (valid_count > 1.0) & valid_mask
    scale = torch.sqrt(
        valid_count / (valid_count - 1.0).clamp_min(1.0)
    )
    response = scale * (signal - mean)
    response = response * usable.to(dtype=response.dtype)
    return response, usable


def _ring_average(signal: torch.Tensor) -> torch.Tensor:
    """Average an 11x11 neighbourhood after excluding its centred 5x5."""
    if signal.ndim != 5:
        raise ValueError("signal must have shape [B, C, T, H, W]")
    batch, channels, time_length, height, width = signal.shape
    frames = signal.permute(0, 2, 1, 3, 4).reshape(
        batch * time_length, channels, height, width
    )
    # Replicate padding keeps a constant image constant at spatial boundaries.
    outer_sum = F.avg_pool2d(
        F.pad(frames, (5, 5, 5, 5), mode="replicate"),
        kernel_size=11,
        stride=1,
    ) * 121.0
    inner_sum = F.avg_pool2d(
        F.pad(frames, (2, 2, 2, 2), mode="replicate"),
        kernel_size=5,
        stride=1,
    ) * 25.0
    ring = (outer_sum - inner_sum) / 96.0
    return ring.reshape(
        batch, time_length, channels, height, width
    ).permute(0, 2, 1, 3, 4)


def _unit_l2_temporal_bandpass(
    signal: torch.Tensor,
    valid_mask: torch.Tensor,
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize mean3-minus-meanK using the actual nested valid windows.

    The squared coefficient norm is ``1/n3 - 1/nK``. Equal valid windows
    carry no bandpass evidence, and padded query frames are always zero.
    """
    short_mean, short_count = _masked_temporal_statistics(
        signal, valid_mask, 3
    )
    long_mean, long_count = _masked_temporal_statistics(
        signal, valid_mask, kernel_size
    )
    # Counts are integer cardinalities; pooling roundoff must not make equal
    # windows appear to have a tiny, nonzero coefficient norm.
    short_count = short_count.round()
    long_count = long_count.round()
    usable = (long_count > short_count) & (short_count > 0) & valid_mask
    norm_squared = (
        short_count.clamp_min(1).reciprocal()
        - long_count.clamp_min(1).reciprocal()
    )
    response = (short_mean - long_mean) / norm_squared.clamp_min(
        torch.finfo(signal.dtype).eps
    ).sqrt()
    return torch.where(usable, response, torch.zeros_like(response)), usable


def _cross_spatial_smooth(signal: torch.Tensor) -> torch.Tensor:
    """Fixed cross filter: centre 1/2, four direct neighbours 1/8 each."""
    if signal.ndim != 5:
        raise ValueError("signal must have shape [B, C, T, H, W]")
    batch, channels, time_length, height, width = signal.shape
    frames = signal.permute(0, 2, 1, 3, 4).reshape(
        batch * time_length, channels, height, width
    )
    padded = F.pad(frames, (1, 1, 1, 1), mode="replicate")
    smoothed = 0.5 * frames + 0.125 * (
        padded[..., :-2, 1:-1] + padded[..., 2:, 1:-1]
        + padded[..., 1:-1, :-2] + padded[..., 1:-1, 2:]
    )
    return smoothed.reshape(
        batch, time_length, channels, height, width
    ).permute(0, 2, 1, 3, 4)


class BCTProAdapter(nn.Module):
    """Static raw-evidence residual adapter applied to post-TPro features."""

    temporal_kernel_sizes = (5, 9, 17)

    def __init__(
        self,
        variant: str,
        channels: int = 32,
        bottleneck_channels: int = 8,
    ) -> None:
        super().__init__()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                "Unsupported BC-TPro variant %r; expected one of %s"
                % (variant, SUPPORTED_VARIANTS)
            )
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channels and bottleneck_channels must be positive")

        self.variant = variant
        self.channels = int(channels)
        self.bottleneck_channels = int(bottleneck_channels)
        if variant == "none":
            self.evidence_fusion = None
            self.residual_projection = None
            return

        evidence_channels = 9 if variant == "center_ring" else 3
        self.evidence_fusion = nn.Sequential(
            nn.Conv3d(
                evidence_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=False),
        )
        self.residual_projection = nn.Conv3d(
            bottleneck_channels, channels, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def _raw_evidence(
        self,
        raw_frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_raw_frames(raw_frames)
        valid_mask = _valid_frame_mask(raw_frames)
        evidence_parts: List[torch.Tensor] = []
        response_masks: List[torch.Tensor] = []
        auxiliary: Dict[str, torch.Tensor] = {}

        if self.variant == "temporal_control":
            for kernel_size in self.temporal_kernel_sizes:
                average, valid_count = _masked_temporal_statistics(
                    raw_frames, valid_mask, kernel_size
                )
                evidence_parts.append(
                    average * valid_mask.to(dtype=average.dtype)
                )
                response_masks.append(valid_count > 0)
        elif self.variant in ("center_multiscale", "center_spatial_smooth"):
            signal = raw_frames
            if self.variant == "center_spatial_smooth":
                signal = _cross_spatial_smooth(raw_frames)
                auxiliary["smoothed_frames"] = signal
            for kernel_size in self.temporal_kernel_sizes:
                response, usable = _unit_l2_zero_dc_response(
                    signal, valid_mask, kernel_size
                )
                evidence_parts.append(response)
                response_masks.append(usable)
        elif self.variant == "temporal_bandpass":
            for kernel_size in self.temporal_kernel_sizes:
                response, usable = _unit_l2_temporal_bandpass(
                    raw_frames, valid_mask, kernel_size
                )
                evidence_parts.append(response)
                response_masks.append(usable)
        elif self.variant in ("center_ring", "center_ring_difference"):
            background = _ring_average(raw_frames)
            auxiliary["ring_background"] = background
            for kernel_size in self.temporal_kernel_sizes:
                centre_response, centre_usable = (
                    _unit_l2_zero_dc_response(
                        raw_frames, valid_mask, kernel_size
                    )
                )
                background_response, background_usable = (
                    _unit_l2_zero_dc_response(
                        background, valid_mask, kernel_size
                    )
                )
                usable = centre_usable & background_usable
                if self.variant == "center_ring":
                    evidence_parts.extend([
                        centre_response,
                        background_response,
                        centre_response - background_response,
                    ])
                    response_masks.extend([usable, usable, usable])
                else:
                    evidence_parts.append(centre_response - background_response)
                    response_masks.append(usable)
        else:
            raise RuntimeError("variant %r has no evidence path" % self.variant)

        evidence = torch.cat(evidence_parts, dim=1)
        evidence_valid_mask = torch.cat(response_masks, dim=1)
        return evidence, evidence_valid_mask, auxiliary

    def prepare_evidence(
        self,
        raw_frames: torch.Tensor,
        return_aux: bool = False,
    ):
        """Fuse full-frame raw evidence to 8 channels before row chunking."""
        _validate_raw_frames(raw_frames)
        if self.variant == "none":
            if return_aux:
                return None, {
                    "evidence": raw_frames.new_empty(
                        (raw_frames.shape[0], 0, *raw_frames.shape[2:])
                    ),
                    "valid_mask": torch.empty(
                        (raw_frames.shape[0], 0, raw_frames.shape[2], 1, 1),
                        dtype=torch.bool,
                        device=raw_frames.device,
                    ),
                }
            return None

        evidence, evidence_valid_mask, auxiliary = self._raw_evidence(
            raw_frames
        )
        fused_evidence = self.evidence_fusion(evidence)
        if not return_aux:
            return fused_evidence
        auxiliary.update({
            "evidence": evidence,
            "valid_mask": evidence_valid_mask,
            "fused_evidence": fused_evidence,
        })
        return fused_evidence, auxiliary

    def apply_fused_evidence(
        self,
        features: torch.Tensor,
        fused_evidence: Optional[torch.Tensor],
        return_delta: bool = False,
    ):
        """Project an already fused (and possibly row-sliced) evidence map."""
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError(
                "features must have shape [B, %d, T, H, W], got %s"
                % (self.channels, tuple(features.shape))
            )
        if self.variant == "none":
            if return_delta:
                return features, None
            return features
        if fused_evidence is None:
            raise ValueError("fused_evidence is required for %s" % self.variant)
        expected = (
            features.shape[0],
            self.bottleneck_channels,
            *features.shape[2:],
        )
        if fused_evidence.shape != expected:
            raise ValueError(
                "fused_evidence has shape %s, expected %s"
                % (tuple(fused_evidence.shape), expected)
            )
        residual_delta = self.residual_projection(fused_evidence)
        output = features + residual_delta
        if return_delta:
            return output, residual_delta
        return output

    def forward(
        self,
        features: torch.Tensor,
        raw_frames: torch.Tensor,
        return_aux: bool = False,
    ):
        if features.shape[0] != raw_frames.shape[0] or (
            features.shape[2:] != raw_frames.shape[2:]
        ):
            raise ValueError(
                "features and raw_frames must share B/T/H/W dimensions"
            )
        if return_aux:
            fused_evidence, auxiliary = self.prepare_evidence(
                raw_frames, return_aux=True
            )
            output, residual_delta = self.apply_fused_evidence(
                features, fused_evidence, return_delta=True
            )
            auxiliary["residual_delta"] = residual_delta
            return output, auxiliary
        fused_evidence = self.prepare_evidence(raw_frames)
        return self.apply_fused_evidence(features, fused_evidence)


def build_bc_tpro_adapter(
    variant: str,
    channels: int = 32,
    bottleneck_channels: int = 8,
) -> BCTProAdapter:
    return BCTProAdapter(
        variant=variant,
        channels=channels,
        bottleneck_channels=bottleneck_channels,
    )


__all__ = [
    "SUPPORTED_VARIANTS",
    "BCTProAdapter",
    "build_bc_tpro_adapter",
    "_valid_frame_mask",
    "_masked_temporal_statistics",
    "_unit_l2_zero_dc_response",
    "_ring_average",
    "_unit_l2_temporal_bandpass",
    "_cross_spatial_smooth",
]
