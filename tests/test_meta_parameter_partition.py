import unittest

import torch

from adaptpro.models.event_mini_dunet import MiniEventDUNet


def _parameter_ids(parameters):
    return {id(parameter) for parameter in parameters}


class MetaParameterPartitionTest(unittest.TestCase):
    def setUp(self):
        self.model = MiniEventDUNet(
            {
                "IlluNet": {"base_chs": 8, "illumination_set": [0]},
                "ImageNet": {
                    "base_chs": 8,
                    "voxel_grid_channel": 4,
                    "snr_threshold_list": [0.6, 0.5],
                },
                "ReferenceBranch": {
                    "enabled": True,
                    "encoder_blocks": 1,
                    "fusion_blocks": 0,
                    "alignment_downsample": 2,
                },
            }
        )

    def test_reference_is_the_only_meta_representation(self):
        meta_ids = _parameter_ids(self.model.meta_parameters())
        reference_ids = _parameter_ids(self.model.reference_parameters())
        adapter_ids = _parameter_ids(self.model.reference_adapter_parameters())
        adapter_task_ids = _parameter_ids(self.model.reference_branch.task_parameters())
        recovery_ids = _parameter_ids(self.model.recovery_parameters())
        backbone_ids = _parameter_ids(self.model.backbone_parameters())
        all_ids = _parameter_ids(self.model.parameters())

        self.assertTrue(meta_ids)
        self.assertEqual(meta_ids, reference_ids)
        self.assertEqual(adapter_ids, meta_ids | adapter_task_ids)
        self.assertTrue(meta_ids < adapter_ids)
        self.assertFalse(meta_ids & adapter_task_ids)
        self.assertTrue(adapter_task_ids <= recovery_ids)
        self.assertEqual(recovery_ids, backbone_ids)
        self.assertFalse(meta_ids & recovery_ids)
        self.assertEqual(all_ids, meta_ids | recovery_ids)

        meta_names = {
            name
            for name, parameter in self.model.named_parameters()
            if id(parameter) in meta_ids
        }
        self.assertTrue(meta_names)
        self.assertTrue(
            all(
                name.startswith("reference_branch.frame_encoder.")
                for name in meta_names
            )
        )

    def test_recovery_reset_does_not_modify_meta_representation(self):
        meta_before = [parameter.detach().clone() for parameter in self.model.meta_parameters()]
        self.model.reset_recovery_parameters()
        meta_after = list(self.model.meta_parameters())

        self.assertEqual(len(meta_before), len(meta_after))
        for before, after in zip(meta_before, meta_after):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
