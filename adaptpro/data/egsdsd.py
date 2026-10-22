"""SDSD loading with EGLOL's reference-frame and paired augmentation protocol."""

import os

import numpy as np
from torch.utils.data import ConcatDataset

from adaptpro.data.eglol import eglol_withNE_dataset


class egsdsd_withNE_dataset(eglol_withNE_dataset):
    """Return the same low/event/gt/img_refer sample interface as EGLOL.

    The inherited methods handle reference selection, crop/voxel alignment,
    normalization, synchronized flips/transpose and reproducible seeded reads.
    SDSD retains its indoor-low / outdoor-normal event acquisition protocol.
    """

    def __init__(
        self,
        dataset_root,
        height,
        width,
        seq_name,
        is_train,
        voxel_grid_channel,
        is_split_event,
        is_indoor,
        reference_first_probability=0.1,
        reference_dropout_probability=0.1,
        geometric_augmentation=False,
    ):
        if not isinstance(is_indoor, bool):
            raise ValueError("is_indoor must be true (indoor) or false (outdoor).")
        # Initialize images/reference settings without requiring low split
        # events: outdoor SDSD uses events from the normal directory instead.
        super().__init__(
            dataset_root=dataset_root,
            height=height,
            width=width,
            seq_name=seq_name,
            is_train=is_train,
            voxel_grid_channel=voxel_grid_channel,
            is_split_event=False,
            reference_first_probability=reference_first_probability,
            reference_dropout_probability=reference_dropout_probability,
            geometric_augmentation=geometric_augmentation,
        )
        self.H = height
        self.W = width
        self.is_split_event = is_split_event
        self.is_indoor = is_indoor

        # Preserve SDSD's existing frame/event slice (first three, last two).
        self.noraml_img_list = self.noraml_img_list[3:-2]
        self.low_img_list = self.low_img_list[3:-2]
        self.num_input = min(len(self.low_img_list), len(self.noraml_img_list))
        if self.is_split_event:
            self.noraml_ev_list = sorted(
                name for name in self.noraml_img_list_all
                if name.lower().endswith(".npz")
                and name.lower() != "normalight_event.npz"
            )[3:-2]
            self.low_ev_list = sorted(
                name for name in self.low_img_list_all
                if name.lower().endswith(".npz")
                and name.lower() != "lowlight_event.npz"
            )[3:-2]
            event_list = self.low_ev_list if self.is_indoor else self.noraml_ev_list
            self.num_input = min(self.num_input, len(event_list))
        if self.num_input == 0:
            raise ValueError(
                "No usable SDSD samples remain after [3:-2] trimming in sequence: "
                f"{os.path.join(dataset_root, seq_name)}"
            )

    def _load_event_input(self, index):
        """Read indoor events from low/ and outdoor events from normal/."""
        if self.is_split_event:
            if self.is_indoor:
                event_path = os.path.join(self.low_img_folder, self.low_ev_list[index])
            else:
                event_path = os.path.join(self.noraml_img_folder, self.noraml_ev_list[index])
        else:
            event_path = self.low_event_file if self.is_indoor else self.normal_event_file

        try:
            with np.load(event_path, allow_pickle=False) as archive:
                if "arr_0" not in archive:
                    raise KeyError("missing `arr_0`")
                events = archive["arr_0"]
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Failed to load event data from {event_path}: {exc}") from exc

        # Match EGLOL's event validation. SDSD archives commonly store events
        # as uint32, which older PyTorch versions cannot pass to
        # torch.from_numpy. Float64 keeps integer timestamps exact throughout
        # the crop step and is supported by every target PyTorch version.
        if events.ndim == 1 and events.dtype.names:
            required_fields = ("timestamp", "x", "y", "polarity")
            missing_fields = [field for field in required_fields if field not in events.dtype.names]
            if missing_fields:
                raise ValueError(f"Event file {event_path} is missing fields: {missing_fields}")
            events = np.stack([events[field] for field in required_fields], axis=1)
        if events.ndim != 2 or events.shape[1] < 4:
            raise ValueError(f"Event file {event_path} must contain an [N, 4] array, got {events.shape}.")
        if len(events) > 1 and np.any(np.diff(events[:, 0]) < 0):
            raise ValueError(f"Event timestamps must be sorted in ascending order: {event_path}")

        self.events = events[:, :4].astype(np.float64, copy=False)
        return self.events if self.is_split_event else self.get_event(index)


def get_egsdsd_withNE_dataset(
    dataset_root,
    center_cropped_height,
    random_cropped_width,
    is_train,
    is_split_event,
    voxel_grid_channel,
    is_indoor,
    reference_first_probability=0.1,
    reference_dropout_probability=0.1,
    geometric_augmentation=False,
):
    """Build SDSD sequences with EGLOL-compatible crop/reference settings."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset split directory does not exist: {dataset_root}")
    seq_dataset_list = []
    for seq in sorted(os.listdir(dataset_root)):
        if os.path.isdir(os.path.join(dataset_root, seq)):
            seq_dataset_list.append(
                egsdsd_withNE_dataset(
                    dataset_root=dataset_root,
                    height=center_cropped_height,
                    width=random_cropped_width,
                    seq_name=seq,
                    is_train=is_train,
                    voxel_grid_channel=voxel_grid_channel,
                    is_split_event=is_split_event,
                    is_indoor=is_indoor,
                    reference_first_probability=reference_first_probability,
                    reference_dropout_probability=reference_dropout_probability,
                    geometric_augmentation=geometric_augmentation,
                )
            )
    if not seq_dataset_list:
        raise ValueError(f"No sequence directories found in dataset split: {dataset_root}")
    return ConcatDataset(seq_dataset_list)
