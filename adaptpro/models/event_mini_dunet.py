from torch import nn

from adaptpro.models.egretinex import IllumiinationNet, ImageEnhanceNet
from adaptpro.models.reference_branch import ReferenceFeatureInjector


class MiniEventDUNet(nn.Module):
    """EvLight with an optional, isolated reference-frame adapter.

    The EvLight ``ImageEnhanceNet`` and ``Unet_ReFormer`` remain the backbone.
    The reference branch has a single residual injection point immediately
    before ``Unet_ReFormer`` and can be removed with one config flag.
    """

    def __init__(self, config):
        super().__init__()
        image_config = dict(config.get("ImageNet") or {})
        if not image_config:
            raise ValueError("MiniEventDUNet requires model.ImageNet configuration.")

        illumination_config = dict(config.get("IlluNet") or {})
        illumination_config.setdefault("base_chs", image_config.get("base_chs", 48))
        illumination_config.setdefault("illumination_set", [0])
        # Keep the official EvLight attribute names so its pretrained weights
        # can be loaded non-strictly with only reference-branch keys missing.
        self.IllumiinationNet = IllumiinationNet(illumination_config)
        self.ImageEnhanceNet = ImageEnhanceNet(image_config)

        reference_config = dict(config.get("ReferenceBranch") or {})
        self.reference_enabled = bool(reference_config.pop("enabled", False))
        if self.reference_enabled:
            reference_config.setdefault("channels", image_config.get("base_chs", 48))
            self.reference_branch = ReferenceFeatureInjector(**reference_config)
        else:
            self.reference_branch = None

    def reset_states(self):
        self.ImageEnhanceNet.reset_states()

    def forward(self, batch):
        local_batch = dict(batch)
        (
            local_batch["illumination"],
            local_batch["illu_feature"],
        ) = self.IllumiinationNet(local_batch)
        return self.ImageEnhanceNet(
            local_batch,
            reference_injector=self.reference_branch,
        )
