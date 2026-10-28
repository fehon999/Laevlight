from pathlib import Path
import tempfile
import unittest

import torch

from adaptpro.models.event_mini_dunet import MiniEventDUNet
from adaptpro.utils import load_checkpoint, save_checkpoint


class PretrainModelCompatibilityTest(unittest.TestCase):
    @staticmethod
    def model_config(mode="reference_affine", enabled=True):
        return {
            "IlluNet": {"base_chs": 8, "illumination_set": [0]},
            "ImageNet": {
                "base_chs": 8,
                "voxel_grid_channel": 4,
                "snr_factor": 3,
                "snr_threshold_list": [0.6, 0.5, 0.4],
            },
            "ReferenceBranch": {
                "enabled": enabled,
                "encoder_blocks": 1,
                "fusion_mode": mode,
                "zero_init": True,
            },
        }

    def test_original_checkpoint_layout_roundtrips_strictly_in_all_modes(self):
        for mode, enabled in (
            ("reference_affine", True),
            ("latent_stats", True),
            ("spatial_align", True),
            ("reference_affine", False),
        ):
            with self.subTest(mode=mode, enabled=enabled):
                config = self.model_config(mode, enabled)
                model = MiniEventDUNet(config)
                original = model.state_dict()
                # These prefixes are the existing EvLight/adapter checkpoint API.
                expected_prefixes = {"IllumiinationNet", "ImageEnhanceNet"}
                if enabled:
                    expected_prefixes.add("reference_branch")
                self.assertEqual({key.split(".")[0] for key in original}, expected_prefixes)
                restored = MiniEventDUNet(config)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pretrain.pt"
                    save_checkpoint(path, model, {"epoch": 3})
                    checkpoint = load_checkpoint(path, restored, strict=True)
                self.assertEqual(checkpoint["epoch"], 3)
                for key, value in restored.state_dict().items():
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_encoder_initialization_remains_equal_and_independent(self):
        model = MiniEventDUNet(self.model_config())
        reference = model.reference_branch.frame_encoder
        current = model.reference_branch.current_encoder
        for reference_parameter, current_parameter in zip(reference.parameters(), current.parameters()):
            torch.testing.assert_close(reference_parameter, current_parameter, rtol=0, atol=0)
            self.assertIsNot(reference_parameter, current_parameter)
            self.assertTrue(reference_parameter.requires_grad)
            self.assertTrue(current_parameter.requires_grad)

    def test_zero_initialized_reference_keeps_evlight_prediction(self):
        torch.manual_seed(31)
        model = MiniEventDUNet(self.model_config()).eval()
        batch = {
            "low": torch.rand(1, 3, 32, 32),
            "event": torch.randn(1, 4, 32, 32),
            "img_refer": torch.rand(1, 3, 32, 32),
            "reference_valid": torch.tensor([True]),
            "ill_list": [torch.rand(1, 1, 32, 32)],
        }
        with torch.no_grad():
            augmented = model(batch)
            model.reference_branch = None
            baseline = model(batch)
        torch.testing.assert_close(augmented, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
