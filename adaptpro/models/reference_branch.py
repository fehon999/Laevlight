"""Lightweight reference-frame latent injection for EvLight.

The reference_affine mode learns bounded contrast/mean corrections to the
backbone from current/reference statistics, without transferring reference
spatial content. Legacy latent_stats and spatial_align modes remain available
for existing checkpoints and ablations. All modes inject once before
``Unet_ReFormer``.
"""

from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualFeatureBlock(nn.Module):
    """Small depthwise residual block used only by the reference adapter."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, feature):
        return feature + self.block(feature)


class SharedFrameEncoder(nn.Module):
    """Encode one RGB frame into the adapter feature space."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            *(ResidualFeatureBlock(channels) for _ in range(num_blocks))
        )

    def forward(self, image):
        return self.body(self.stem(image))


class ReferenceFeatureInjector(nn.Module):
    """Fuse current/reference latents and inject a bounded residual feature.

    ``reference_affine`` conditions a bounded, channel-wise affine correction
    of the backbone on current/reference statistics, using a single gate.
    ``latent_stats`` (the constructor default for old configs) preserves the normalized current-frame latent
    and its channel standard deviation while replacing its channel mean with
    the reference mean.  A learned channel gate decides how strongly to use
    that transfer.  ``spatial_align`` keeps the former flow-aligned fusion for
    controlled ablations.
    """

    def __init__(
        self,
        channels: int,
        encoder_blocks: int = 2,
        fusion_blocks: int = 0,
        fusion_mode: str = "latent_stats",
        residual_scale: float = 0.1,
        statistics_eps: float = 1e-6,
        alignment_downsample: int = 4,
        max_offset: float = 16.0,
        use_alignment: bool = False,
        zero_init: bool = True,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if fusion_mode not in {"latent_stats", "spatial_align", "reference_affine"}:
            raise ValueError(
                "fusion_mode must be 'latent_stats', 'spatial_align', or 'reference_affine', "
                f"got {fusion_mode!r}."
            )
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError(
                f"residual_scale must be between 0 and 1, got {residual_scale}."
            )
        if statistics_eps <= 0:
            raise ValueError(
                f"statistics_eps must be positive, got {statistics_eps}."
            )
        if alignment_downsample <= 0:
            raise ValueError(
                "alignment_downsample must be positive, "
                f"got {alignment_downsample}."
            )
        if max_offset < 0:
            raise ValueError(f"max_offset must be non-negative, got {max_offset}.")

        self.channels = int(channels)
        self.fusion_mode = fusion_mode
        self.residual_scale = float(residual_scale)
        self.statistics_eps = float(statistics_eps)
        self.alignment_downsample = int(alignment_downsample)
        self.max_offset = float(max_offset)
        self.use_alignment = bool(use_alignment)
        self.zero_init = bool(zero_init)

        # Only frame_encoder (the reference-frame extractor) is omega/meta.
        # current_encoder has an identical initialization but independent task
        # parameters, so no current-frame processing leaks into omega.
        self.frame_encoder = SharedFrameEncoder(self.channels, int(encoder_blocks))
        self.current_encoder = SharedFrameEncoder(self.channels, int(encoder_blocks))
        self.current_encoder.load_state_dict(self.frame_encoder.state_dict())

        hidden_channels = max(self.channels // 2, 8)
        if self.fusion_mode == "spatial_align":
            alignment_channels = self.channels * 3
            self.flow_predictor = nn.Sequential(
                nn.Conv2d(alignment_channels, self.channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 2, 3, padding=1),
            )
            nn.init.zeros_(self.flow_predictor[-1].weight)
            nn.init.zeros_(self.flow_predictor[-1].bias)

            # Current, aligned reference, absolute difference, and product.
            fusion_channels = self.channels * 4
            self.gate = nn.Sequential(
                nn.Conv2d(fusion_channels, self.channels, 1),
                nn.GELU(),
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    groups=self.channels,
                ),
                nn.GELU(),
                nn.Conv2d(self.channels, 1, 1),
            )
            self.fusion_in = nn.Sequential(
                nn.Conv2d(fusion_channels, self.channels, 1),
                nn.GELU(),
            )
            self.fusion_body = nn.Sequential(
                *(
                    ResidualFeatureBlock(self.channels)
                    for _ in range(int(fusion_blocks))
                )
            )
            self.fusion_out = nn.Conv2d(self.channels, self.channels, 1)
        else:
            self.flow_predictor = None
            # Per-channel current/reference mean and standard deviation.
            statistics_channels = self.channels * 4
            self.gate = nn.Sequential(
                nn.Conv2d(statistics_channels, self.channels, 1),
                nn.GELU(),
                nn.Conv2d(self.channels, self.channels, 1),
            )
            if self.fusion_mode == "reference_affine":
                # Learn the relation between both encoders' statistics rather
                # than directly subtracting independently learned channels.
                self.fusion_in = nn.Sequential(
                    nn.Conv2d(statistics_channels, self.channels, 1),
                    nn.GELU(),
                )
                self.fusion_body = nn.Identity()
                self.fusion_out = nn.Conv2d(self.channels, self.channels * 2, 1)
            else:
                # Preserve the original mode and its checkpoint shapes.
                fusion_channels = self.channels * 2
                self.fusion_in = nn.Identity()
                self.fusion_body = nn.Identity()
                self.fusion_out = nn.Conv2d(fusion_channels, self.channels, 1)
        # Begin conservatively even when zero_init is disabled.
        nn.init.constant_(
            self.gate[-1].bias, 0.0 if self.fusion_mode == "reference_affine" else -1.0
        )
        if self.zero_init:
            nn.init.zeros_(self.fusion_out.weight)
            nn.init.zeros_(self.fusion_out.bias)

    def reference_feature_parameters(self):
        """Return only the reference-frame feature extractor (omega)."""

        return self.frame_encoder.parameters()

    def task_parameters(self):
        """Return current encoding, fusion, gating, and injection parameters."""

        task_modules = [
            self.current_encoder,
            self.gate,
            self.fusion_in,
            self.fusion_body,
            self.fusion_out,
        ]
        if self.flow_predictor is not None:
            task_modules.insert(1, self.flow_predictor)
        return chain.from_iterable(
            module.parameters() for module in task_modules
        )

    def reset_task_parameters(self):
        """Reset task modules without modifying the reference feature extractor."""

        task_modules = [
            self.current_encoder,
            self.gate,
            self.fusion_in,
            self.fusion_body,
            self.fusion_out,
        ]
        if self.flow_predictor is not None:
            task_modules.insert(1, self.flow_predictor)
        for root_module in task_modules:
            for module in root_module.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        # Start each task adaptation in the same feature space as the learned
        # reference meta encoder, while keeping current_encoder trainable theta.
        self.current_encoder.load_state_dict(self.frame_encoder.state_dict())
        if self.flow_predictor is not None:
            nn.init.zeros_(self.flow_predictor[-1].weight)
            nn.init.zeros_(self.flow_predictor[-1].bias)
        nn.init.constant_(
            self.gate[-1].bias, 0.0 if self.fusion_mode == "reference_affine" else -1.0
        )
        if self.zero_init:
            nn.init.zeros_(self.fusion_out.weight)
            nn.init.zeros_(self.fusion_out.bias)

    @staticmethod
    def _warp(feature, flow):
        """Warp ``feature`` with a pixel-space backward flow field."""

        batch, _channels, height, width = feature.shape
        dtype = feature.dtype
        device = feature.device
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        sample_grid = base_grid + flow.permute(0, 2, 3, 1)

        if width > 1:
            sample_grid_x = 2.0 * sample_grid[..., 0] / (width - 1) - 1.0
        else:
            sample_grid_x = torch.zeros_like(sample_grid[..., 0])
        if height > 1:
            sample_grid_y = 2.0 * sample_grid[..., 1] / (height - 1) - 1.0
        else:
            sample_grid_y = torch.zeros_like(sample_grid[..., 1])
        normalized_grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)
        return F.grid_sample(
            feature,
            normalized_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _align(self, current_feature, reference_feature):
        if self.flow_predictor is None:
            raise RuntimeError("Alignment is only available in spatial_align mode.")
        if not self.use_alignment or self.max_offset == 0:
            flow = reference_feature.new_zeros(
                reference_feature.shape[0],
                2,
                reference_feature.shape[2],
                reference_feature.shape[3],
            )
            return reference_feature, flow

        height, width = current_feature.shape[-2:]
        low_height = max(1, height // self.alignment_downsample)
        low_width = max(1, width // self.alignment_downsample)
        current_low = F.adaptive_avg_pool2d(current_feature, (low_height, low_width))
        reference_low = F.adaptive_avg_pool2d(reference_feature, (low_height, low_width))
        alignment_input = torch.cat(
            (current_low, reference_low, torch.abs(current_low - reference_low)),
            dim=1,
        )
        low_max_offset = self.max_offset / self.alignment_downsample
        flow_low = torch.tanh(self.flow_predictor(alignment_input)) * low_max_offset
        flow = F.interpolate(
            flow_low,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ) * self.alignment_downsample
        return self._warp(reference_feature, flow), flow

    @staticmethod
    def _channel_statistics(feature, eps):
        """Return stable per-channel spatial mean/std in feature dtype."""

        feature_float = feature.float()
        mean = feature_float.mean(dim=(2, 3), keepdim=True)
        variance = (feature_float - mean).square().mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(variance + eps)
        return mean.to(feature.dtype), std.to(feature.dtype)

    def _reference_mean_transfer(self, current_feature, reference_feature):
        """Keep current structure/std and replace only its channel mean."""

        current_mean, current_std = self._channel_statistics(
            current_feature,
            self.statistics_eps,
        )
        reference_mean, reference_std = self._channel_statistics(
            reference_feature,
            self.statistics_eps,
        )
        normalized_current = (current_feature - current_mean) / current_std
        transferred = normalized_current * current_std + reference_mean
        return transferred, (
            current_mean,
            current_std,
            reference_mean,
            reference_std,
        )

    def _latent_statistics_forward(
        self,
        backbone_feature,
        current_feature,
        reference_feature,
        valid_mask,
    ):
        transferred, statistics = self._reference_mean_transfer(
            current_feature,
            reference_feature,
        )
        statistics_input = torch.cat(statistics, dim=1)
        confidence = torch.sigmoid(self.gate(statistics_input)) * valid_mask

        mixed_feature = current_feature + confidence * (
            transferred - current_feature
        )
        fusion_input = torch.cat((current_feature, mixed_feature), dim=1)
        residual = self.fusion_out(
            self.fusion_body(self.fusion_in(fusion_input))
        )
        return backbone_feature + self.residual_scale * confidence * residual

    def _reference_affine_forward(
        self,
        backbone_feature,
        current_feature,
        reference_feature,
        valid_mask,
    ):
        current_statistics = self._channel_statistics(current_feature, self.statistics_eps)
        reference_statistics = self._channel_statistics(reference_feature, self.statistics_eps)
        statistics = torch.cat((*current_statistics, *reference_statistics), dim=1)
        confidence = torch.sigmoid(self.gate(statistics)) * valid_mask
        affine = self.fusion_out(self.fusion_body(self.fusion_in(statistics)))
        contrast, shift = torch.tanh(affine).chunk(2, dim=1)

        # Only the backbone supplies spatial content. Scaling the shift by its
        # std keeps corrections relative to each channel's feature magnitude.
        # |delta| <= residual_scale * (|feature - mean| + std), with one gate.
        # FP32 avoids unstable statistics/cancellation under autocast.
        with torch.autocast(device_type=backbone_feature.device.type, enabled=False):
            feature = backbone_feature.float()
            mean, std = self._channel_statistics(feature, self.statistics_eps)
            residual = contrast.float() * (feature - mean) + shift.float() * std
            output = feature + self.residual_scale * confidence.float() * residual
        return output.to(backbone_feature.dtype)

    def _spatial_alignment_forward(
        self,
        backbone_feature,
        current_feature,
        reference_feature,
        valid_mask,
    ):
        aligned_reference, _flow = self._align(
            current_feature,
            reference_feature,
        )
        difference = torch.abs(current_feature - aligned_reference)
        interaction = current_feature * aligned_reference
        fusion_input = torch.cat(
            (backbone_feature, aligned_reference, difference, interaction),
            dim=1,
        )
        confidence = torch.sigmoid(self.gate(fusion_input)) * valid_mask
        residual = self.fusion_out(
            self.fusion_body(self.fusion_in(fusion_input))
        )
        return backbone_feature + self.residual_scale * confidence * residual

    @staticmethod
    def _validity_mask(reference_valid, reference_image):
        batch = reference_image.shape[0]
        if reference_valid is None:
            # Backward compatibility for datasets produced before the explicit
            # validity key was introduced.
            reference_valid = reference_image.detach().abs().flatten(1).sum(dim=1) > 0
        if not torch.is_tensor(reference_valid):
            reference_valid = torch.as_tensor(reference_valid, device=reference_image.device)
        reference_valid = reference_valid.to(
            device=reference_image.device,
            dtype=reference_image.dtype,
        )
        if reference_valid.numel() != batch:
            raise ValueError(
                "reference_valid must contain one value per batch item; "
                f"got {reference_valid.numel()} for batch size {batch}."
            )
        return reference_valid.reshape(batch, 1, 1, 1)

    def forward(
        self,
        backbone_feature,
        current_image,
        reference_image,
        reference_valid=None,
    ):
        if backbone_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} backbone channels, "
                f"got {backbone_feature.shape[1]}."
            )
        if current_image.shape != reference_image.shape:
            raise ValueError(
                "Current and reference images must have the same shape, got "
                f"{tuple(current_image.shape)} and {tuple(reference_image.shape)}."
            )
        if (
            backbone_feature.shape[0] != current_image.shape[0]
            or backbone_feature.shape[-2:] != current_image.shape[-2:]
        ):
            raise ValueError(
                "Backbone feature and current image must share batch/spatial dimensions, "
                f"got {tuple(backbone_feature.shape)} and {tuple(current_image.shape)}."
            )

        current_feature = self.current_encoder(current_image)
        reference_feature = self.frame_encoder(reference_image)
        valid_mask = self._validity_mask(
            reference_valid,
            reference_image,
        ).to(dtype=current_feature.dtype)
        if self.fusion_mode == "reference_affine":
            return self._reference_affine_forward(
                backbone_feature, current_feature, reference_feature, valid_mask
            )
        if self.fusion_mode == "latent_stats":
            return self._latent_statistics_forward(
                backbone_feature,
                current_feature,
                reference_feature,
                valid_mask,
            )
        return self._spatial_alignment_forward(
            backbone_feature,
            current_feature,
            reference_feature,
            valid_mask,
        )
