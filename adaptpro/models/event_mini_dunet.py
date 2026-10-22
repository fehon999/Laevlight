from itertools import chain

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

    def shared_parameters(self):
        return iter(())

    def task_parameters(self):
        """Return EvLight plus non-meta adapter parameters (theta)."""

        parameter_groups = [
            self.IllumiinationNet.parameters(),
            self.ImageEnhanceNet.parameters(),
        ]
        if self.reference_branch is not None:
            parameter_groups.append(self.reference_branch.task_parameters())
        return chain(*parameter_groups)

    def backbone_parameters(self):
        """Backward-compatible alias for the complete task parameter group."""

        return self.task_parameters()

    def reference_parameters(self):
        """Return only reference-frame feature-extractor parameters."""

        if self.reference_branch is None:
            return iter(())
        return self.reference_branch.reference_feature_parameters()

    def reference_adapter_parameters(self):
        """Return the complete adapter for diagnostics, not as a meta group."""

        if self.reference_branch is None:
            return iter(())
        return self.reference_branch.parameters()

    def meta_parameters(self):
        """Return reference feature extraction as the sole meta representation.

        Omega excludes current-frame encoding, statistics/alignment, gating, fusion, and
        all EvLight weights.
        """

        return self.reference_parameters()

    def degradation_parameters(self):
        """Backward-compatible alias for legacy bilevel training code."""

        return self.meta_parameters()

    def recovery_parameters(self):
        """Return EvLight and reference-fusion task modules as theta."""

        return self.task_parameters()

    def reset_recovery_parameters(self):
        """Reset theta without modifying the learned reference feature extractor."""

        for root_module in (self.IllumiinationNet, self.ImageEnhanceNet):
            for module in root_module.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        if self.reference_branch is not None:
            self.reference_branch.reset_task_parameters()

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
