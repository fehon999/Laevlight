from copy import deepcopy

from adaptpro.models.event_mini_dunet import MiniEventDUNet


def get_model(config, event_channels=None):
    """Build the configured model from the repository's dictionary config."""

    model_config = deepcopy(config.get("model", config))
    name = model_config.get("name", "MiniEventDUNet")
    if event_channels is not None:
        model_config.setdefault("ImageNet", {})["voxel_grid_channel"] = int(
            event_channels
        )
    if name == "MiniEventDUNet":
        return MiniEventDUNet(model_config)
    raise ValueError(f"Model {name} is not supported.")


__all__ = ["MiniEventDUNet", "get_model"]
