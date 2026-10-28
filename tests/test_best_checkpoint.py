from pathlib import Path
import tempfile
import unittest

import torch

from adaptpro.training import save_progress


class BestCheckpointTest(unittest.TestCase):
    def test_best_weights_follow_ssim_even_when_psnr_disagrees(self):
        model = torch.nn.Linear(1, 1, bias=False)
        cfg = {"output_dir": "unused"}
        best_ssim = float("-inf")
        # Higher PSNR alone must not replace best.pt; tied SSIM keeps the
        # earlier checkpoint. Latest weights must still advance every epoch.
        epochs = (
            (20.0, 0.8, 1),
            (30.0, 0.7, 1),
            (10.0, 0.9, 3),
            (99.0, 0.9, 3),
        )
        with tempfile.TemporaryDirectory() as directory:
            for epoch, (psnr, ssim, expected_best_epoch) in enumerate(epochs, 1):
                with self.subTest(epoch=epoch):
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    best_ssim = save_progress(
                        directory, model, cfg, "pretrain_all", epoch,
                        {"psnr": psnr, "ssim_metric": ssim}, best_ssim,
                    )
                    best = torch.load(Path(directory) / "best.pt", map_location="cpu")
                    latest = torch.load(Path(directory) / "pretrain_all_latest.pt", map_location="cpu")
                    self.assertEqual(best["epoch"], expected_best_epoch)
                    self.assertEqual(best["best_ssim"], best_ssim)
                    self.assertNotIn("best_psnr", best)
                    self.assertEqual(best["model"]["weight"].item(), expected_best_epoch)
                    self.assertEqual(latest["epoch"], epoch)
                    self.assertEqual(latest["model"]["weight"].item(), epoch)
                    self.assertEqual(latest["test"]["ssim_metric"], ssim)

    def test_first_epoch_is_saved_even_with_negative_ssim(self):
        with tempfile.TemporaryDirectory() as directory:
            score = save_progress(
                directory, torch.nn.Linear(1, 1), {}, "pretrain_all", 1,
                {"ssim_metric": -0.2}, float("-inf"),
            )
            self.assertEqual(score, -0.2)
            self.assertTrue((Path(directory) / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
