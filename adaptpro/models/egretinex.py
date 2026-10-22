"""EvLight backbone modules adapted to the local batch/config interfaces."""

import torch
import torch.nn.functional as F
from torch import nn

from adaptpro.models.base_block.Trans import Unet_ReFormer


def _cfg_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class IllumiinationNet(nn.Module):
    """Estimate the initial light-up map used by the original EvLight model."""

    def __init__(self, config):
        super().__init__()
        illumination_set = _cfg_get(
            config,
            "illumination_set",
            _cfg_get(config, "illumiantion_set", [0]),
        )
        self.illumination_set = [int(index) for index in illumination_set]
        if not self.illumination_set:
            raise ValueError("illumination_set must select at least one illumination map.")

        base_channels = int(_cfg_get(config, "base_chs", 48))
        legacy_level = int(
            _cfg_get(
                config,
                "illumination_level",
                _cfg_get(config, "illumiantion_level", len(self.illumination_set)),
            )
        )
        hidden_channels = int(_cfg_get(config, "hidden_chs", legacy_level * 2))
        input_channels = 3 + len(self.illumination_set)
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(hidden_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, batch):
        if "ill_list" not in batch:
            raise KeyError("EvLight illumination estimation requires batch['ill_list'].")
        if max(self.illumination_set) >= len(batch["ill_list"]):
            raise ValueError(
                "illumination_set references an unavailable illumination map: "
                f"indices={self.illumination_set}, available={len(batch['ill_list'])}."
            )
        initial_illumination = torch.cat(
            [batch["ill_list"][index] for index in self.illumination_set],
            dim=1,
        )
        illumination_feature = self.ill_extractor(
            torch.cat((initial_illumination, batch["low"]), dim=1)
        )
        illumination = self.reduce(illumination_feature)
        return illumination, illumination_feature


class ImageEnhanceNet(nn.Module):
    """Original EvLight image-event restoration backbone.

    ``reference_injector`` is optional and is the only extension point added to
    the backbone.  When it is ``None`` the computation is the baseline EvLight
    path.
    """

    def __init__(self, config):
        super().__init__()
        self.base_chs = int(_cfg_get(config, "base_chs", 48))
        self.snr_factor = float(_cfg_get(config, "snr_factor", 3.0))
        voxel_grid_channel = int(_cfg_get(config, "voxel_grid_channel", 32))

        self.ev_img_align = nn.Conv2d(self.base_chs * 2, self.base_chs, 1)
        self.ev_extractor = nn.Sequential(
            nn.Conv2d(voxel_grid_channel, self.base_chs, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.img_extractor = nn.Sequential(
            nn.Conv2d(3, self.base_chs, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Keep the original EvLight constructor arguments unchanged so
        # existing backbone checkpoints retain the same topology.
        self.Unet_ReFormer = Unet_ReFormer(
            dim=self.base_chs,
            snr_threshold_list=list(
                _cfg_get(config, "snr_threshold_list", [0.5, 0.5, 0.5])
            ),
            use_rnn=bool(_cfg_get(config, "use_rnn", False)),
        )

    def reset_states(self):
        self.Unet_ReFormer.reset_states()

    def _snr_generate(self, low_image, blurred_image):
        dark = (
            low_image[:, 0:1] * 0.299
            + low_image[:, 1:2] * 0.587
            + low_image[:, 2:3] * 0.114
        )
        light = (
            blurred_image[:, 0:1] * 0.299
            + blurred_image[:, 1:2] * 0.587
            + blurred_image[:, 2:3] * 0.114
        )
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 1e-4).contiguous()
        mask_max = mask.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
        return torch.clamp(mask * self.snr_factor / (mask_max + 1e-4), 0.0, 1.0).float()

    @staticmethod
    def _blur_if_missing(image):
        return F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)

    def forward(self, batch, reference_injector=None):
        low_image = batch["low"]
        illumination = batch["illumination"]
        low_blur = batch.get("lowlight_image_blur")
        if low_blur is None:
            low_blur = self._blur_if_missing(low_image)

        light_up_image = low_image * illumination + low_image
        light_up_blur = low_blur * illumination + low_blur
        snr_light_up = self._snr_generate(light_up_image, light_up_blur)

        event_feature = self.ev_extractor(batch["event"])
        image_feature = self.img_extractor(light_up_image)
        holistic_feature = self.ev_img_align(
            torch.cat((event_feature, image_feature), dim=1)
        )

        if reference_injector is not None:
            if "img_refer" not in batch:
                raise KeyError(
                    "The reference branch is enabled but batch['img_refer'] is missing."
                )
            holistic_feature = reference_injector(
                backbone_feature=holistic_feature,
                current_image=light_up_image,
                reference_image=batch["img_refer"],
                reference_valid=batch.get("reference_valid"),
            )

        return self.Unet_ReFormer(
            holistic_feature,
            light_up_image,
            image_feature,
            snr_light_up.detach(),
            event_feature,
        )


class EgLlie(nn.Module):
    """Compatibility wrapper around the official EvLight modules."""

    def __init__(self, config):
        super().__init__()
        illumination_config = _cfg_get(config, "IlluNet", {})
        image_config = _cfg_get(config, "ImageNet", {})
        self.IllumiinationNet = IllumiinationNet(illumination_config)
        self.ImageEnhanceNet = ImageEnhanceNet(image_config)

    def reset_states(self):
        self.ImageEnhanceNet.reset_states()

    def forward(self, batch):
        local_batch = dict(batch)
        local_batch["illumination"], local_batch["illu_feature"] = self.IllumiinationNet(
            local_batch
        )
        return self.ImageEnhanceNet(local_batch)
