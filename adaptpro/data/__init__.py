"""Dataset construction for single-stage training and independent inference."""

from os.path import join

from adaptpro.data.eglol import get_eglol_withNE_dataset
from adaptpro.data.egsdsd import get_egsdsd_withNE_dataset


def get_split_dataset(config, split="test", is_train=False):
    """Build only the requested split using the existing SDE/SDSD protocol.

    SDE uses the EGLOL reader for both indoor and outdoor data. SDSD selects
    its event source with ``is_indoor``: low/ for indoor, normal/ for outdoor.
    Evaluation keeps reference selection and center cropping deterministic.
    """
    if config is None:
        raise ValueError("Missing `data` section in config.")

    name = config.get("name")
    dataset_factories = {
        "eglol": get_eglol_withNE_dataset,
        "sdsd": get_egsdsd_withNE_dataset,
        "get_eglol_withNE_dataset": get_eglol_withNE_dataset,
        "get_egsdsd_withNE_dataset": get_egsdsd_withNE_dataset,
    }
    if name not in dataset_factories:
        raise ValueError(f"Unknown dataset: {name}")

    required = ("root", "img_height", "img_width", "voxel_grid_channel")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing data config keys: {missing}")

    dataset_kwargs = {}
    if name in {"sdsd", "get_egsdsd_withNE_dataset"}:
        if not isinstance(config.get("is_indoor"), bool):
            raise ValueError("SDSD requires data.is_indoor to be true (indoor) or false (outdoor).")
        dataset_kwargs["is_indoor"] = config["is_indoor"]

    return dataset_factories[name](
        dataset_root=join(config["root"], split),
        center_cropped_height=config["img_height"],
        random_cropped_width=config["img_width"],
        is_train=is_train,
        geometric_augmentation=is_train and config.get("geometric_augmentation", False),
        is_split_event=config.get("is_split_event", True),
        voxel_grid_channel=config["voxel_grid_channel"],
        reference_first_probability=config.get("reference_first_probability", 0.1),
        reference_dropout_probability=config.get("reference_dropout_probability", 0.1),
        **dataset_kwargs,
    )


def get_dataset(config):
    """Return training and test datasets; no auxiliary split is required."""
    if config is None:
        raise ValueError("Missing `data` section in config.")
    train_set = get_split_dataset(config, config.get("train_split", "train"), is_train=True)
    test_set = get_split_dataset(config, config.get("test_split", "test"), is_train=False)
    return train_set, test_set


__all__ = ["get_dataset", "get_split_dataset"]
