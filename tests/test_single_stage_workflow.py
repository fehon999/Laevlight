"""Exercise real dataset readers, training, checkpoints and test-only inference."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

import infer_event_dunet
import train_event_dunet
from adaptpro.models import get_model


class SingleStageWorkflowTest(unittest.TestCase):
    def test_optional_diagnostics_allow_all_preserved_reference_modes(self):
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/sde_in.yaml").read_text(encoding="utf-8")
        )
        cfg["model"]["IlluNet"]["base_chs"] = 8
        cfg["model"]["ImageNet"].update(base_chs=8, voxel_grid_channel=4)
        cfg["model"]["ReferenceBranch"]["encoder_blocks"] = 1
        batch = {
            "low": torch.rand(1, 3, 32, 32), "event": torch.rand(1, 4, 32, 32),
            "gt": torch.rand(1, 3, 32, 32), "img_refer": torch.rand(1, 3, 32, 32),
            "reference_valid": torch.tensor([True]), "ill_list": [torch.rand(1, 1, 32, 32)],
        }
        for mode in ("reference_affine", "latent_stats", "spatial_align", "disabled"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                cfg["model"]["ReferenceBranch"].update(
                    enabled=mode != "disabled", fusion_mode=mode if mode != "disabled" else "reference_affine",
                )
                model = get_model(cfg, event_channels=4)
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(io.StringIO()):
                    infer_event_dunet.run_inference(
                        model, [batch], torch.device("cpu"), directory, print_reference_diagnostics=True,
                    )
                self.assertEqual(len(list(Path(directory).rglob("*.png"))), 1)
                if mode == "reference_affine":
                    self.assertIn("reference: valid=1/1", output.getvalue())
                else:
                    self.assertIn("Reference diagnostics skipped", output.getvalue())

    def test_all_four_datasets_train_and_infer_without_an_auxiliary_split(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for dataset_id in ("sde_in", "sde_out", "sdsd_in", "sdsd_out"):
            with self.subTest(dataset_id=dataset_id), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cfg = yaml.safe_load((config_dir / f"{dataset_id}.yaml").read_text(encoding="utf-8"))
                cfg["device"] = "cpu"
                cfg["output_dir"] = str(root / "checkpoints")
                cfg["data"].update(root=str(root / "data"), img_height=32, img_width=32, voxel_grid_channel=4)
                cfg["model"]["event_channels"] = 4
                cfg["model"]["IlluNet"]["base_chs"] = 8
                cfg["model"]["ImageNet"].update(base_chs=8, voxel_grid_channel=4)
                cfg["model"]["ReferenceBranch"]["encoder_blocks"] = 1
                cfg["loader"].update(batch_size=2, num_workers=0)
                cfg["train"]["epochs"] = 1
                # The smoke test does not download LPIPS or pretrained weights.
                cfg["loss"]["lpips_weight"] = 0.0
                cfg["inference"].update(
                    checkpoint=str(root / "checkpoints" / "best.pt"),
                    output_dir=str(root / "predictions"), print_reference_diagnostics=False,
                )
                self.write_dataset(Path(cfg["data"]["root"]), dataset_id)
                config_path = root / "config.yaml"
                config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with patch("sys.argv", ["train_event_dunet.py", "--config", str(config_path)]):
                        train_event_dunet.main()
                    for name in ("pretrain_all_latest.pt", "best.pt", "global_last.pt"):
                        self.assertTrue((root / "checkpoints" / name).is_file())
                    checkpoint = torch.load(root / "checkpoints" / "pretrain_all_latest.pt", map_location="cpu")
                    self.assertEqual(checkpoint["stage"], "pretrain_all")
                    self.assertEqual(checkpoint["epoch"], 1)
                    self.assertEqual(checkpoint["config"]["data"]["dataset_id"], dataset_id)
                    self.assertTrue(np.isfinite(checkpoint["test"]["total"]))
                    # Inference must work with test data alone.
                    train_dir = root / "data" / "train"
                    train_dir.rename(root / "data" / "unused_train")
                    with patch("sys.argv", ["infer_event_dunet.py", "--config", str(config_path)]):
                        infer_event_dunet.main()
                self.assertEqual(len(list((root / "predictions").rglob("*.png"))), 2)

    @staticmethod
    def write_dataset(root, dataset_id):
        rng = np.random.default_rng(23)
        # SDSD's existing reader trims three initial and two final frames.
        frames = 7 if dataset_id.startswith("sdsd") else 2
        event_folder = "normal" if dataset_id == "sdsd_out" else "low"
        for split in ("train", "test"):
            seq = root / split / "sequence"
            for folder in ("normal", "low"):
                (seq / folder).mkdir(parents=True)
            for index in range(frames):
                gt = rng.integers(30, 240, size=(36, 36, 3), dtype=np.uint8)
                cv2.imwrite(str(seq / "normal" / f"{index:04d}.png"), gt)
                cv2.imwrite(str(seq / "low" / f"{index:04d}.png"), gt // 4)
                events = np.array([[0, 10, 10, 1], [1, 20, 20, -1]], dtype=np.float64)
                np.savez(seq / event_folder / f"{index:04d}.npz", events)


if __name__ == "__main__":
    unittest.main()
