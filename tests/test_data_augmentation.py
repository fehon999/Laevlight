import itertools
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from adaptpro.data import get_dataset
from adaptpro.data.eglol import eglol_withNE_dataset


class DataAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.dataset = object.__new__(eglol_withNE_dataset)
        self.dataset.is_train = True
        self.dataset.geometric_augmentation = True

    def sample(self, height=3, width=3):
        pattern = torch.arange(height * width).reshape(1, height, width).float()
        sample = {
            key: pattern.repeat(channels, 1, 1) + offset
            for key, channels, offset in (
                ("low", 3, 0), ("gt", 3, 10), ("img_refer", 3, 20),
                ("lowlight_image_blur", 3, 30), ("event", 4, -40),
            )
        }
        # Distinct time bins and signed event values must retain channel order.
        sample["event"] += torch.arange(4).reshape(4, 1, 1) * 50
        sample.update(ill_list=[pattern + 50, pattern + 60],
                      reference_valid=torch.tensor(True), seq_name="seq", frame_id="0")
        return sample

    def test_all_eight_transforms_keep_modalities_and_time_bins_aligned(self):
        for flips in itertools.product((False, True), repeat=3):
            with self.subTest(flips=flips):
                sample = self.sample()
                rng = Mock()
                rng.random.side_effect = [0.1 if enabled else 0.9 for enabled in flips]
                augmented = self.dataset._augment_sample(sample, rng=rng)
                # Index the original spatial grid to check the full mapping.
                indices = np.arange(9).reshape(3, 3)
                if flips[0]:
                    indices = np.fliplr(indices)
                if flips[1]:
                    indices = np.flipud(indices)
                if flips[2]:
                    indices = indices.T
                indices = torch.tensor(indices.copy()).flatten()
                pairs = [(sample[key], augmented[key]) for key in
                         ("low", "gt", "img_refer", "event", "lowlight_image_blur")]
                pairs += list(zip(sample["ill_list"], augmented["ill_list"]))
                for before, after in pairs:
                    expected = before.flatten(1)[:, indices].reshape_as(before)
                    self.assertTrue(torch.equal(after, expected))
                    self.assertTrue(after.is_contiguous())
                self.assertIs(augmented["reference_valid"], sample["reference_valid"])
                self.assertEqual(augmented["frame_id"], "0")

    def test_rectangular_crop_keeps_shape_and_empty_reference(self):
        sample = self.sample(2, 3)
        sample["img_refer"].zero_()
        sample["reference_valid"] = torch.tensor(False)
        augmented = self.dataset._augment_sample(sample, rng=Mock(random=Mock(return_value=0.1)))
        self.assertEqual(augmented["low"].shape, (3, 2, 3))
        self.assertTrue(torch.equal(augmented["low"], sample["low"].flip(-1, -2)))
        self.assertEqual(augmented["img_refer"].count_nonzero().item(), 0)
        self.assertFalse(augmented["reference_valid"].item())

    def test_evaluation_and_disabled_augmentation_do_not_draw_randomness(self):
        for is_train, enabled in ((False, True), (True, False)):
            self.dataset.is_train = is_train
            self.dataset.geometric_augmentation = enabled
            rng = Mock()
            sample = self.sample()
            self.assertIs(self.dataset._augment_sample(sample, rng=rng), sample)
            rng.random.assert_not_called()

    def test_seeded_loading_is_reproducible_and_both_entrypoints_augment(self):
        with tempfile.TemporaryDirectory() as root:
            seq = Path(root) / "seq"
            for folder in ("normal", "low"):
                (seq / folder).mkdir(parents=True)
                for index in range(3):
                    image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3) + index
                    cv2.imwrite(str(seq / folder / f"{index}.png"), image)
            for index in range(3):
                np.savez(seq / "low" / f"{index}.npz",
                         np.array([[0, 2, 2, 1], [1, 3, 3, -1]], dtype=np.float32))
            dataset = eglol_withNE_dataset(root, 4, 4, "seq", True, 4, True,
                                          geometric_augmentation=True)
            with patch.object(dataset, "_augment_sample", wraps=dataset._augment_sample) as augment:
                state = random.getstate()
                first = dataset.getitem_with_seed(1, 17)
                second = dataset.getitem_with_seed(1, 17)
                self.assertEqual(random.getstate(), state)
                for key in ("low", "gt", "event", "img_refer", "lowlight_image_blur"):
                    self.assertTrue(torch.equal(first[key], second[key]))
                self.assertTrue(torch.equal(first["ill_list"][0], second["ill_list"][0]))
                dataset[1]
                self.assertEqual(augment.call_count, 3)

    def test_config_enables_only_training_split(self):
        with patch("adaptpro.data.get_eglol_withNE_dataset") as factory:
            get_dataset(dict(name="get_eglol_withNE_dataset", root="unused", img_height=4,
                             img_width=4, voxel_grid_channel=4, geometric_augmentation=True))
        train, test = [call.kwargs for call in factory.call_args_list]
        self.assertTrue(train["geometric_augmentation"])
        for evaluation in (test,):
            self.assertFalse(evaluation["is_train"])
            self.assertFalse(evaluation.get("geometric_augmentation", False))


if __name__ == "__main__":
    unittest.main()
