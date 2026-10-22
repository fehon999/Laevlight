import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
import cv2
import random
import torch


class eglol_withNE_dataset(Dataset):
    def __init__(
            self,
            #./train/O_99
            dataset_root,
            height,
            width,
            seq_name,
            is_train,
            voxel_grid_channel,
            is_split_event,
            reference_first_probability=0.1,
            reference_dropout_probability=0.1,
            geometric_augmentation=False,
    ):
        self.noraml_img_folder = os.path.join(dataset_root, seq_name, "normal")
        self.noraml_img_list_all = os.listdir(self.noraml_img_folder)
        self.noraml_img_list = sorted(
            name for name in self.noraml_img_list_all if name.lower().endswith(".png")
        )

        self.low_img_folder = os.path.join(dataset_root, seq_name, "low")
        self.low_img_list_all = os.listdir(self.low_img_folder)
        self.low_img_list = sorted(
            name for name in self.low_img_list_all if name.lower().endswith(".png")
        )

        self.is_split_event = is_split_event
        if self.is_split_event:
            # use split event file
            self.noraml_ev_list = sorted(
                name
                for name in self.noraml_img_list_all
                if name.lower().endswith(".npz") and name.lower() != "normalight_event.npz"
            )
            self.low_ev_list = sorted(
                name
                for name in self.low_img_list_all
                if name.lower().endswith(".npz") and name.lower() != "lowlight_event.npz"
            )
        else:
            # use the whole event file
            self.normal_event_file = os.path.join(
                dataset_root, seq_name, "normal", "normalight_event.npz"
            )
            self.low_event_file = os.path.join(
                dataset_root, seq_name, "low", "lowlight_event.npz"
            )

        self.num_input = min(len(self.low_img_list), len(self.noraml_img_list))
        if self.is_split_event:
            self.num_input = min(self.num_input, len(self.low_ev_list))
        if self.num_input == 0:
            raise ValueError(f"No usable samples found in sequence: {os.path.join(dataset_root, seq_name)}")
        self.ev_idx = None
        self.events = None
        self.center_cropped_height = height
        self.random_cropped_width = width
        self.is_train = is_train
        self.geometric_augmentation = bool(geometric_augmentation)
        self.seq_name = seq_name
        self.voxel_grid_channel = voxel_grid_channel
        if not 0.0 <= reference_first_probability <= 1.0:
            raise ValueError(
                "reference_first_probability must be between 0 and 1, "
                f"got {reference_first_probability}."
            )
        if not 0.0 <= reference_dropout_probability <= 1.0:
            raise ValueError(
                "reference_dropout_probability must be between 0 and 1, "
                f"got {reference_dropout_probability}."
            )
        if reference_first_probability + reference_dropout_probability > 1.0:
            raise ValueError(
                "reference_first_probability + reference_dropout_probability "
                "must not exceed 1."
            )
        self.reference_first_probability = reference_first_probability
        self.reference_dropout_probability = reference_dropout_probability

    def __len__(self):
        return self.num_input

    def _empty_voxel_grid(self):
        return torch.zeros(
            (self.voxel_grid_channel, self.center_cropped_height, self.random_cropped_width),
            dtype=torch.float32,
        )

    def get_event(self, idx):
        """Split single event from the whole event file """
        if self.events is None or len(self.events) == 0:
            return np.empty((0, 4), dtype=np.float32)

        # if there is no previous frame, set start_t as the timestamp of first event
        if idx == 0:
            start_t = self.events[0, 0]
        else:
            start_t = int(self.low_img_list[idx - 1][:-4])

        # if there is no next frame, set end_t as the timestamp of last event
        if idx == self.num_input - 1:
            end_t = self.events[-1, 0]
        else:
            end_t = int(self.low_img_list[idx + 1][:-4])

        timestamps = self.events[:, 0]
        start_side = "left" if idx == 0 else "right"
        end_side = "right" if idx == self.num_input - 1 else "left"
        ev_start_idx = np.searchsorted(timestamps, start_t, side=start_side)
        ev_end_idx = np.searchsorted(timestamps, end_t, side=end_side)
        event = self.events[ev_start_idx:ev_end_idx]

        return event

    def _load_event_input(self, index):
        if self.is_split_event:
            event_path = os.path.join(self.low_img_folder, self.low_ev_list[index])
        else:
            event_path = self.low_event_file

        try:
            with np.load(event_path, allow_pickle=False) as archive:
                if "arr_0" not in archive:
                    raise KeyError("missing `arr_0`")
                events = archive["arr_0"]
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Failed to load event data from {event_path}: {exc}") from exc

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

        self.events = events[:, :4]
        return self.events if self.is_split_event else self.get_event(index)

    def _crop(self, input_frame_list, events_list, rng=None):
        """crop frame and events
           training: random crop
           testing: central crop
        """
        if not input_frame_list:
            raise ValueError("Cannot crop an empty image list.")
        frame_height, frame_width = input_frame_list[0].shape[:2]
        crop_height = self.center_cropped_height
        crop_width = self.random_cropped_width
        if crop_height > frame_height or crop_width > frame_width:
            raise ValueError(
                f"Crop size {(crop_height, crop_width)} exceeds image size "
                f"{(frame_height, frame_width)} in sequence {self.seq_name}."
            )
        for frame in input_frame_list[1:]:
            if frame.shape[:2] != (frame_height, frame_width):
                raise ValueError(
                    f"All paired images must share the same size; got "
                    f"{(frame_height, frame_width)} and {frame.shape[:2]} in sequence {self.seq_name}."
                )

        rng = rng or random
        if self.is_train:
            min_y = rng.randint(0, frame_width - crop_width)
            min_x = rng.randint(0, frame_height - crop_height)
        else:
            min_y = (frame_width - crop_width) // 2
            min_x = (frame_height - crop_height) // 2
        max_y = min_y + crop_width
        max_x = min_x + crop_height
        crop_image_list = []
        for input_frame in input_frame_list:
            input_frames = input_frame[min_x:max_x, min_y:max_y, :]
            input_frames_torch = torch.from_numpy(input_frames).permute(2, 0, 1).float()
            crop_image_list.append(input_frames_torch)

        output_events_list = []
        for events in events_list:
            mask_x = torch.where(
                (events[:, 2] < max_x) & (events[:, 2] >= min_x)
            )
            event_x = torch.index_select(events, 0, mask_x[0])
            mask_y = torch.where(
                (event_x[:, 1] < max_y) & (event_x[:, 1] >= min_y)
            )
            event_y = torch.index_select(event_x, 0, mask_y[0])
            event = event_y.clone()
            event[:, 2] = event_y[:, 2] - min_x
            event[:, 1] = event_y[:, 1] - min_y
            output_events_list.append(event)
        return crop_image_list, output_events_list

    @staticmethod
    def _read_rgb(path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _select_reference_index(self, index, rng=None):
        """Select a same-sequence reference without using the current GT frame."""
        num_references = len(self.noraml_img_list)
        valid_indices = [
            reference_index
            for reference_index in range(num_references)
            if reference_index != index
        ]
        if not valid_indices:
            return None

        # Evaluation must be deterministic. Keep the original first-frame
        # protocol except when the current sample itself is the first frame.
        if not self.is_train:
            return 0 if index != 0 else valid_indices[0]

        rng = rng or random
        selection = rng.random()
        if selection < self.reference_dropout_probability:
            return None
        if (
            index != 0
            and selection
            < self.reference_dropout_probability + self.reference_first_probability
        ):
            return 0

        # Randomly use any other normal-light frame in the same sequence.
        # Excluding frame 0 here preserves the configured first-frame ratio.
        random_indices = [
            reference_index
            for reference_index in valid_indices
            if reference_index != 0
        ]
        candidates = random_indices or valid_indices
        return rng.choice(candidates)

    def _load_reference(self, index, gt_image, rng=None):
        reference_index = self._select_reference_index(index, rng=rng)
        if reference_index is None:
            # A one-frame sequence has no non-target reference. Use an empty
            # reference instead of leaking the ground-truth image.
            return np.zeros_like(gt_image), False
        return (
            self._read_rgb(
                os.path.join(
                    self.noraml_img_folder,
                    self.noraml_img_list[reference_index],
                )
            ),
            True,
        )

    def _illumiantion_map(self, img):
        """obtain illumination prior"""
        intial_illumiantion_map = np.max(img, axis=2)
        return intial_illumiantion_map

    def _generate_voxel_grid(self, event):
        """obtain voxel grid """
        width, height = self.random_cropped_width, self.center_cropped_height
        if event.numel() == 0:
            return torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)

        event_start = event[0, 0]
        event_end = event[-1, 0]
        duration = (event_end - event_start).clamp_min(1e-6)

        ch = (
                (event[:, 0].to(torch.float32) - event_start)
                / duration
                * self.voxel_grid_channel
        ).long()
        torch.clamp_(ch, 0, self.voxel_grid_channel - 1)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)
        ep[ep == 0] = -1

        voxel_grid = torch.zeros(
            (self.voxel_grid_channel, height, width), dtype=torch.float32
        )
        voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)

        return voxel_grid

    def _augment_sample(self, sample, rng=None):
        """Apply one spatial transform to all paired images, priors and voxel bins."""
        if not self.is_train or not self.geometric_augmentation:
            return sample
        rng = rng or random
        flip_h, flip_v = rng.random() < 0.5, rng.random() < 0.5
        # Rectangular crops must keep their shape for batching.
        transpose = sample["low"].shape[-2] == sample["low"].shape[-1] and rng.random() < 0.5

        def transform(tensor):
            if flip_h:
                tensor = tensor.flip(-1)
            if flip_v:
                tensor = tensor.flip(-2)
            if transpose:
                tensor = tensor.transpose(-2, -1)
            return tensor.contiguous()

        augmented = dict(sample)
        for key in ("low", "gt", "img_refer", "event", "lowlight_image_blur"):
            augmented[key] = transform(sample[key])
        augmented["ill_list"] = [transform(prior) for prior in sample["ill_list"]]
        return augmented

    def __getitem__(self, index):
        # 1. load event
        event_input = self._load_event_input(index)

        # 2. load image & obtain illumination map prior
        img_low = self._read_rgb(os.path.join(self.low_img_folder, self.low_img_list[index]))
        img_blur = cv2.blur(img_low, (5, 5))
        img_low_illumination_map = self._illumiantion_map(img_low)
        img_gt = self._read_rgb(os.path.join(self.noraml_img_folder, self.noraml_img_list[index]))
        img_refer, reference_valid = self._load_reference(index, img_gt)
        # h, w -> h, w, c
        img_low_illumination_map = np.expand_dims(
            img_low_illumination_map / 255, axis=-1
        )

        event_input_torch = torch.from_numpy(event_input)

        crop_img_list, crop_event_list = self._crop(
            [
                img_low,
                img_gt,
                img_low_illumination_map,
                img_blur,
                img_refer
            ],
            [event_input_torch],
        )

        input_voxel_grid_list = []
        # obtain event voxel grid
        for crop_event in crop_event_list:
            if crop_event.numel() == 0:
                input_voxel_grid = self._empty_voxel_grid()
            else:
                crop_event[:, 0] = crop_event[:, 0] - crop_event[0, 0]
                input_voxel_grid = self._generate_voxel_grid(crop_event)
            input_voxel_grid_list.append(input_voxel_grid)

        del (
            event_input,
            event_input_torch,
            crop_event_list,
        )
        # img_low = crop_img_list[0].permute(1, 2, 0).numpy()
        # img_gt=crop_img_list[1].permute(1, 2, 0).numpy()
        # img_refer=crop_img_list[4].permute(1, 2, 0).numpy()
        #
        # img_low = cv2.cvtColor(img_low, cv2.COLOR_RGB2BGR)
        # img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
        # img_refer = cv2.cvtColor(img_refer, cv2.COLOR_RGB2BGR)
        # img_all = np.hstack((img_low, img_gt, img_refer))
        # os.makedirs("./save", exist_ok=True)
        # cv2.imwrite(f"./save/imgf{index}.png", img_all)
        # print("00000000000000000000")
        sample = {
            "low": crop_img_list[0] / 255,
            "gt": crop_img_list[1] / 255,
            "event": input_voxel_grid_list[0],
            "lowlight_image_blur": crop_img_list[3] / 255,
            "ill_list": [
                crop_img_list[2],
            ],
            #----------------
            "img_refer": crop_img_list[4]/255,
            "reference_valid": torch.tensor(reference_valid, dtype=torch.bool),
            "seq_name": self.seq_name,
            "frame_id": self.low_img_list[index].split(".")[0],
        }

        # reduce memory coast
        self.events = None

        return self._augment_sample(sample)

    def getitem_with_seed(self, index, seed):
        """For video sequence data loading, using specified random seed to maintain data augmentation consistency across sequence frames"""
        # 1. load event
        event_input = self._load_event_input(index)

        # 2. load image & obtain illumination map prior
        img_low = self._read_rgb(os.path.join(self.low_img_folder, self.low_img_list[index]))
        img_blur = cv2.blur(img_low, (5, 5))
        img_low_illumination_map = self._illumiantion_map(img_low)
        img_gt = self._read_rgb(os.path.join(self.noraml_img_folder, self.noraml_img_list[index]))
        reference_rng = random.Random(seed)
        img_refer, reference_valid = self._load_reference(
            index,
            img_gt,
            rng=reference_rng,
        )
        # h, w -> h, w, c
        img_low_illumination_map = np.expand_dims(
            img_low_illumination_map / 255, axis=-1
        )

        event_input_torch = torch.from_numpy(event_input)

        crop_img_list, crop_event_list = self._crop(
            [
                img_low,
                img_gt,
                img_low_illumination_map,
                img_blur,
                img_refer
            ],
            [event_input_torch],
            rng=random.Random(seed),
        )

        input_voxel_grid_list = []
        # obtain event voxel grid
        for crop_event in crop_event_list:
            if crop_event.numel() == 0:
                input_voxel_grid = self._empty_voxel_grid()
            else:
                crop_event[:, 0] = crop_event[:, 0] - crop_event[0, 0]
                input_voxel_grid = self._generate_voxel_grid(crop_event)
            input_voxel_grid_list.append(input_voxel_grid)

        del (
            event_input,
            event_input_torch,
            crop_event_list,
        )


        sample = {
            "low": crop_img_list[0] / 255,
            "gt": crop_img_list[1] / 255,
            "event": input_voxel_grid_list[0],
            "lowlight_image_blur": crop_img_list[3] / 255,
            "ill_list": [
                crop_img_list[2],
            ],
            "img_refer": crop_img_list[4]/255,
            "reference_valid": torch.tensor(reference_valid, dtype=torch.bool),
            "seq_name": self.seq_name,
            "frame_id": self.low_img_list[index].split(".")[0],
        }

        # reduce memory coast
        self.events = None

        return self._augment_sample(sample, rng=random.Random(seed))


def get_eglol_withNE_dataset(
        #/home/user/dufeihan/single_Dataset/SDE_Dataset/SDE_dataset/SDE_outdoor/sde_out_release/sde_out_release/train
        dataset_root,
        center_cropped_height,
        random_cropped_width,
        is_train,
        is_split_event,
        voxel_grid_channel,
        reference_first_probability=0.1,
        reference_dropout_probability=0.1,
        geometric_augmentation=False,
):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset split directory does not exist: {dataset_root}")
    all_seqs = os.listdir(dataset_root)
    all_seqs.sort()

    seq_dataset_list = []

    for seq in all_seqs:
        if os.path.isdir(os.path.join(dataset_root, seq)):
            # load each sequence one by one
            seq_dataset_list.append(
                eglol_withNE_dataset(
                    dataset_root,
                    center_cropped_height,
                    random_cropped_width,
                    seq,
                    is_train,
                    voxel_grid_channel,
                    is_split_event,
                    reference_first_probability,
                    reference_dropout_probability,
                    geometric_augmentation,
                )
            )
    if not seq_dataset_list:
        raise ValueError(f"No sequence directories found in dataset split: {dataset_root}")
    return ConcatDataset(seq_dataset_list)


