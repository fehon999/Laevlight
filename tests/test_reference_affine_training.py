import unittest

import torch

from adaptpro.losses import EnhancementLoss
from adaptpro.models.event_mini_dunet import MiniEventDUNet
from adaptpro.models.reference_branch import ReferenceFeatureInjector


class ReferenceAffineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.feature = torch.randn(2, 8, 16, 16)
        self.current = torch.rand(2, 3, 16, 16)
        self.reference = torch.rand(2, 3, 16, 16)

    def injector(self, **kwargs):
        return ReferenceFeatureInjector(
            channels=8, encoder_blocks=1, fusion_mode="reference_affine", **kwargs
        )

    def test_zero_initialization_and_reset_preserve_backbone(self):
        injector = self.injector()
        args = (self.feature, self.current, self.reference, torch.tensor([True, True]))
        self.assertTrue(torch.equal(injector(*args), self.feature))
        with torch.no_grad():
            injector.fusion_out.bias.fill_(0.8)
        self.assertFalse(torch.equal(injector(*args), self.feature))
        reference_before = [p.detach().clone() for p in injector.reference_feature_parameters()]
        injector.reset_task_parameters()
        self.assertTrue(torch.equal(injector(*args), self.feature))
        for before, after in zip(reference_before, injector.reference_feature_parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_invalid_reference_has_no_output_or_gradient_contribution(self):
        injector = self.injector(zero_init=False)
        reference = self.reference.clone().requires_grad_()
        output = injector(self.feature, self.current, reference, torch.tensor([False, True]))
        self.assertTrue(torch.equal(output[0], self.feature[0]))
        self.assertFalse(torch.equal(output[1], self.feature[1]))
        output.square().sum().backward()
        self.assertEqual(reference.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(reference.grad[1].abs().sum().item(), 0.0)

    def test_reference_content_changes_output_and_zero_scale_disables_it(self):
        injector = self.injector(zero_init=False)
        valid = torch.tensor([True, True])
        original = injector(self.feature, self.current, self.reference, valid)
        changed = injector(self.feature, self.current, self.reference + 0.5, valid)
        self.assertGreater((original - changed).abs().max().item(), 1e-7)
        injector.residual_scale = 0.0
        disabled = injector(self.feature, self.current, self.reference, valid)
        self.assertTrue(torch.equal(disabled, self.feature))

    def test_correction_is_bounded_even_with_saturated_prediction(self):
        injector = self.injector(residual_scale=0.1, zero_init=False)
        with torch.no_grad():
            injector.fusion_out.weight.zero_()
            injector.fusion_out.bias.fill_(100.0)
        output = injector(self.feature, self.current, self.reference, [True, True])
        mean, std = injector._channel_statistics(self.feature, injector.statistics_eps)
        bound = 0.1 * ((self.feature - mean).abs() + std)
        self.assertTrue(torch.all((output - self.feature).abs() <= bound + 1e-6))

    def test_autocast_is_finite_on_constant_features(self):
        injector = self.injector(zero_init=False)
        feature = torch.ones_like(self.feature, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = injector(feature, self.current, self.reference, [True, True])
        self.assertEqual(output.dtype, feature.dtype)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())


class StructuralLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.target = torch.rand(2, 3, 16, 16)

    def test_ssim_loss_is_zero_for_identical_images_and_has_useful_gradients(self):
        criterion = EnhancementLoss(lpips_weight=0.0, ssim_weight=0.1)
        self.assertAlmostEqual(criterion.ssim_loss(self.target, self.target).item(), 0.0, places=6)
        pred = (self.target * 0.6 + 0.2).requires_grad_()
        loss = criterion.ssim_loss(pred, self.target)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)
        improved = pred.detach() - pred.grad * 0.1
        self.assertLess(criterion.ssim_loss(improved, self.target).item(), loss.item())

    def test_phi_psi_weights_and_disabled_loss_are_backward_compatible(self):
        pred = self.target * 0.5
        legacy = EnhancementLoss(lpips_weight=0.0)
        old_loss, old_logs = legacy(pred, self.target)
        self.assertIsNone(legacy.ssim)
        self.assertNotIn("phi_ssim", old_logs)
        self.assertTrue(torch.equal(old_loss, legacy.charbonnier_loss(pred, self.target)))
        criterion = EnhancementLoss(
            lpips_weight=0.0, ssim_weight=0.1, psi_ssim_weight=0.2
        )
        phi, logs = criterion.phi_forward(pred, self.target)
        psi, _ = criterion.psi_forward(pred, self.target)
        self.assertIn("phi_ssim", logs)
        torch.testing.assert_close(phi - old_loss, 0.1 * criterion.ssim_loss(pred, self.target))
        torch.testing.assert_close(psi - phi, 0.1 * criterion.ssim_loss(pred, self.target))

    def test_ssim_uses_fp32_even_under_autocast(self):
        criterion = EnhancementLoss(lpips_weight=0.0, ssim_weight=0.1)
        pred = (self.target * 0.7).to(torch.bfloat16).requires_grad_()
        expected = criterion.ssim_loss(pred.float(), self.target)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = criterion.ssim_loss(pred, self.target)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_structural_weights_are_rejected(self):
        for name in ("ssim_weight", "psi_ssim_weight"):
            for value in (-0.1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    EnhancementLoss(lpips_weight=0.0, **{name: value})


class JointTrainingTest(unittest.TestCase):
    def test_one_optimizer_trains_backbone_and_both_encoders(self):
        torch.manual_seed(29)
        model = MiniEventDUNet({
            "IlluNet": {"base_chs": 8, "illumination_set": [0]},
            "ImageNet": {
                "base_chs": 8, "voxel_grid_channel": 4,
                "snr_factor": 3, "snr_threshold_list": [0.6, 0.5, 0.4],
            },
            "ReferenceBranch": {
                "enabled": True, "encoder_blocks": 1,
                "fusion_mode": "reference_affine", "zero_init": True,
            },
        })
        criterion = EnhancementLoss(lpips_weight=0.0, ssim_weight=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        batch = {
            "low": torch.rand(2, 3, 32, 32),
            "event": torch.randn(2, 4, 32, 32),
            "img_refer": torch.rand(2, 3, 32, 32),
            "reference_valid": torch.tensor([True, True]),
            "ill_list": [torch.rand(2, 1, 32, 32)],
        }
        target = torch.rand(2, 3, 32, 32)
        # The zero-initialized last projection learns on step one. All upstream
        # adapter modules must receive task gradients on the next joint step.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            self.assertEqual(pred.shape, target.shape)
            loss, _ = criterion(pred, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()

        for module in (
            model.IllumiinationNet, model.ImageEnhanceNet,
            model.reference_branch.current_encoder, model.reference_branch.frame_encoder,
            model.reference_branch.fusion_in, model.reference_branch.gate,
        ):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)
        meta_ids = {id(p) for p in model.meta_parameters()}
        task_ids = {id(p) for p in model.task_parameters()}
        self.assertFalse(meta_ids & task_ids)
        self.assertEqual(meta_ids | task_ids, {id(p) for p in model.parameters()})
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
