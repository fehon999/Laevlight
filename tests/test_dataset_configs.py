import os
from pathlib import Path
import unittest
from unittest.mock import patch, sentinel

import yaml

from adaptpro.data import get_dataset, get_split_dataset


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DATASETS = ("sde_in", "sde_out", "sdsd_in", "sdsd_out")


class DatasetConfigurationTest(unittest.TestCase):
    def load_config(self, dataset_id):
        with (CONFIG_DIR / f"{dataset_id}.yaml").open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_four_configs_route_to_their_own_dataset_and_event_protocol(self):
        roots, outputs, inference_outputs = set(), set(), set()
        for dataset_id in DATASETS:
            with self.subTest(dataset_id=dataset_id):
                cfg = self.load_config(dataset_id)
                data = cfg["data"]
                self.assertEqual(data["dataset_id"], dataset_id)
                self.assertEqual(Path(data["root"]).name, dataset_id)
                self.assertNotIn("root_pre", data)
                self.assertNotIn("root_meta", data)
                self.assertNotIn("query_split", data)
                self.assertNotIn("stages", cfg["train"])
                roots.add(data["root"])
                outputs.add(cfg["output_dir"])
                inference_outputs.add(cfg["inference"]["output_dir"])
                self.assertEqual(
                    Path(cfg["inference"]["checkpoint"]),
                    Path(cfg["output_dir"]) / "best.pt",
                )
                with patch("adaptpro.data.get_eglol_withNE_dataset") as sde_factory, patch(
                    "adaptpro.data.get_egsdsd_withNE_dataset"
                ) as sdsd_factory:
                    get_dataset(data)
                if dataset_id.startswith("sdsd"):
                    sde_factory.assert_not_called()
                    calls = sdsd_factory.call_args_list
                    for call in calls:
                        self.assertIs(call.kwargs["is_indoor"], dataset_id == "sdsd_in")
                else:
                    sdsd_factory.assert_not_called()
                    calls = sde_factory.call_args_list
                    self.assertTrue(all("is_indoor" not in call.kwargs for call in calls))
                self.assertEqual(len(calls), 2)
                train, test = [call.kwargs for call in calls]
                self.assertEqual(train["dataset_root"], os.path.join(data["root"], "train"))
                self.assertEqual(test["dataset_root"], os.path.join(data["root"], "test"))
                self.assertTrue(train["is_train"])
                self.assertTrue(train["geometric_augmentation"])
                self.assertFalse(test["is_train"])
                self.assertFalse(test["geometric_augmentation"])
        self.assertEqual(len(roots), 4)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(inference_outputs), 4)

    def test_inference_builds_only_requested_split_without_training_augmentation(self):
        data = self.load_config("sde_in")["data"]
        with patch("adaptpro.data.get_eglol_withNE_dataset", return_value=sentinel.test_set) as factory:
            dataset = get_split_dataset(data, split="held_out")
        self.assertIs(dataset, sentinel.test_set)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["dataset_root"], os.path.join(data["root"], "held_out"))
        self.assertFalse(factory.call_args.kwargs["is_train"])
        self.assertFalse(factory.call_args.kwargs["geometric_augmentation"])

    def test_train_and_test_split_names_are_configurable(self):
        data = self.load_config("sde_out")["data"]
        data.update(train_split="train_images", test_split="test_images")
        with patch("adaptpro.data.get_eglol_withNE_dataset", side_effect=[sentinel.train_set, sentinel.test_set]) as factory:
            self.assertEqual(get_dataset(data), (sentinel.train_set, sentinel.test_set))
        self.assertEqual(
            [call.kwargs["dataset_root"] for call in factory.call_args_list],
            [os.path.join(data["root"], split) for split in ("train_images", "test_images")],
        )

    def test_sdsd_requires_an_explicit_boolean_event_source_selector(self):
        for value in (None, "false", 0, 1):
            with self.subTest(value=value):
                data = self.load_config("sdsd_out")["data"]
                data["is_indoor"] = value
                with self.assertRaisesRegex(ValueError, "is_indoor"):
                    get_split_dataset(data)

    def test_missing_root_is_reported_before_opening_dataset_files(self):
        data = self.load_config("sde_in")["data"]
        del data["root"]
        with self.assertRaisesRegex(ValueError, "root"):
            get_dataset(data)

    def test_new_configs_preserve_the_original_pretraining_settings(self):
        baseline = self.load_config("sde_in")
        for dataset_id in DATASETS:
            with self.subTest(dataset_id=dataset_id):
                cfg = self.load_config(dataset_id)
                for key in ("model", "loss", "train", "loader"):
                    self.assertEqual(cfg[key], baseline[key])
                self.assertEqual(cfg["train"]["epochs"], 50)
                self.assertEqual(cfg["train"]["lr"], 0.00015)
                self.assertEqual(cfg["train"]["lr_scheduler"], "cosine")
                self.assertEqual(cfg["train"]["min_lr"], 1e-8)
                self.assertEqual(cfg["train"]["weight_decay"], 0)
                self.assertEqual(cfg["loss"]["ssim_weight"], 0)
                self.assertEqual(cfg["data"]["reference_dropout_probability"], 0)
                self.assertEqual(cfg["model"]["ImageNet"]["snr_threshold_list"], [0.6, 0.5, 0.4])


if __name__ == "__main__":
    unittest.main()
