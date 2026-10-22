from os.path import join
from adaptpro.data.eglol import get_eglol_withNE_dataset
from adaptpro.data.egsdsd import get_egsdsd_withNE_dataset

def get_dataset(config, include_query=True):
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
    dataset_factory = dataset_factories[name]

    required = ("root", "img_height", "img_width", "voxel_grid_channel")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing data config keys: {missing}")

    dataset_kwargs = {}
    if name in {"sdsd", "get_egsdsd_withNE_dataset"}:
        if not isinstance(config.get("is_indoor"), bool):
            raise ValueError("SDSD requires data.is_indoor to be true (indoor) or false (outdoor).")
        dataset_kwargs["is_indoor"] = config["is_indoor"]

    train_split = config.get("train_split", "train")
    query_split = config.get("query_split", config.get("validate_split", config.get("val_split", "validate")))
    test_split = config.get("test_split", "test")

    train_set = dataset_factory(
        dataset_root=join(config.get("root"), train_split),
        center_cropped_height=config.get("img_height"),
        random_cropped_width=config.get("img_width"),
        is_train=True,
        geometric_augmentation=config.get("geometric_augmentation", False),
        is_split_event=config.get("is_split_event", True),
        voxel_grid_channel=config.get("voxel_grid_channel"),
        reference_first_probability=config.get("reference_first_probability", 0.1),
        reference_dropout_probability=config.get("reference_dropout_probability", 0.1),
        **dataset_kwargs,
    )
    query_set = None
    if include_query:
        query_set = dataset_factory(
            dataset_root=join(config.get("root"), query_split),
            center_cropped_height=config.get("img_height"),
            random_cropped_width=config.get("img_width"),
            is_train=False,
            is_split_event=config.get("is_split_event", True),
            voxel_grid_channel=config.get("voxel_grid_channel"),
            reference_first_probability=config.get("reference_first_probability", 0.1),
            reference_dropout_probability=config.get("reference_dropout_probability", 0.1),
            **dataset_kwargs,
        )
    test_set = dataset_factory(
        dataset_root=join(config.get("root"), test_split),
        center_cropped_height=config.get("img_height"),
        random_cropped_width=config.get("img_width"),
        is_train=False,
        is_split_event=config.get("is_split_event", True),
        voxel_grid_channel=config.get("voxel_grid_channel"),
        reference_first_probability=config.get("reference_first_probability", 0.1),
        reference_dropout_probability=config.get("reference_dropout_probability", 0.1),
        **dataset_kwargs,
    )

    return train_set, query_set, test_set


# __all__ = ["get_dataset"]
