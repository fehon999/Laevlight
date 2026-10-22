import unittest

import torch

from adaptpro.models.reference_branch import ReferenceFeatureInjector


class ReferenceFeatureInjectorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.feature = torch.randn(2, 8, 16, 16)
        self.current = torch.rand(2, 3, 16, 16)
        self.reference = torch.rand(2, 3, 16, 16)

    def test_zero_initialized_adapter_preserves_backbone_exactly(self):
        injector = ReferenceFeatureInjector(
            channels=8,
            encoder_blocks=1,
            fusion_blocks=1,
            alignment_downsample=4,
            zero_init=True,
        )
        output = injector(
            self.feature,
            self.current,
            self.reference,
            torch.tensor([True, True]),
        )
        self.assertTrue(torch.equal(output, self.feature))

    def test_invalid_reference_is_fully_suppressed(self):
        injector = ReferenceFeatureInjector(
            channels=8,
            encoder_blocks=1,
            fusion_blocks=0,
            use_alignment=False,
            zero_init=False,
        )
        with torch.no_grad():
            injector.fusion_out.weight.zero_()
            injector.fusion_out.bias.fill_(1.0)

        output = injector(
            self.feature,
            self.current,
            self.reference,
            torch.tensor([False, True]),
        )
        self.assertTrue(torch.equal(output[0], self.feature[0]))
        self.assertFalse(torch.equal(output[1], self.feature[1]))

    def test_alignment_output_shape_matches_backbone(self):
        injector = ReferenceFeatureInjector(
            channels=8,
            encoder_blocks=1,
            fusion_blocks=0,
            fusion_mode="spatial_align",
            alignment_downsample=3,
            max_offset=8.0,
            use_alignment=True,
        )
        output = injector(
            self.feature,
            self.current,
            self.reference,
            torch.tensor([True, True]),
        )
        self.assertEqual(output.shape, self.feature.shape)

    def test_latent_statistics_transfer_uses_reference_mean_and_current_std(self):
        injector = ReferenceFeatureInjector(
            channels=8,
            encoder_blocks=1,
            fusion_mode="latent_stats",
        )
        current_feature = torch.randn(2, 8, 9, 7)
        reference_feature = torch.randn(2, 8, 9, 7) + 3.0

        transferred, _statistics = injector._reference_mean_transfer(
            current_feature,
            reference_feature,
        )

        self.assertTrue(
            torch.allclose(
                transferred.mean(dim=(2, 3)),
                reference_feature.mean(dim=(2, 3)),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                transferred.std(dim=(2, 3), unbiased=False),
                current_feature.std(dim=(2, 3), unbiased=False),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_latent_statistics_uses_single_linear_projection_without_flow(self):
        injector = ReferenceFeatureInjector(
            channels=8,
            encoder_blocks=1,
            fusion_mode="latent_stats",
        )
        self.assertIsNone(injector.flow_predictor)
        self.assertIsInstance(injector.fusion_in, torch.nn.Identity)
        self.assertIsInstance(injector.fusion_body, torch.nn.Identity)
        self.assertEqual(injector.fusion_out.in_channels, 16)
        self.assertEqual(injector.fusion_out.out_channels, 8)

    def test_unknown_fusion_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceFeatureInjector(channels=8, fusion_mode="unknown")

if __name__ == "__main__":
    unittest.main()
