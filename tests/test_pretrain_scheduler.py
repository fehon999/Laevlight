from contextlib import ExitStack
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from adaptpro.training import make_optimizer, make_scheduler, read_config
import train_event_dunet


class PretrainSchedulerTest(unittest.TestCase):
    def test_default_training_loop_uses_configured_cosine(self):
        cfg = read_config(Path(__file__).resolve().parents[1] / "configs/sde_in.yaml")
        stage = cfg["train"]
        epochs = stage["epochs"]
        self.assertGreater(epochs, 0)
        model = torch.nn.Linear(1, 1)
        used_rates = []
        optimizers = []

        def train_epoch(model, loader, criterion, optimizer, *args, **kwargs):
            used_rates.append(optimizer.param_groups[0]["lr"])
            optimizers.append(optimizer)
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            return {"total": 0.0}

        with tempfile.TemporaryDirectory() as output_dir, ExitStack() as stack:
            replacements = {
                "read_config": lambda _: cfg,
                "prepare_runtime": lambda *a, **kw: (
                    torch.device("cpu"), [1], [1], model, None, output_dir),
                "make_grad_scaler": lambda *a: None,
                "train_standard_epoch": train_epoch,
                "validate": lambda *a, **kw: {"psnr": 1.0},
                "save_progress": lambda *a, **kw: 1.0,
                "save_checkpoint": lambda *a, **kw: None,
                "log_line": lambda *a, **kw: None,
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(train_event_dunet, name, replacement))
            stack.enter_context(patch("sys.argv", ["train_event_dunet.py", "--config", "configs/sde_in.yaml"]))
            stack.enter_context(patch("builtins.print"))
            train_event_dunet.main()
        self.assertEqual(len(used_rates), epochs)
        self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 1)
        for epoch, actual in enumerate(used_rates):
            expected = stage["min_lr"] + (stage["lr"] - stage["min_lr"]) * (
                1 + math.cos(math.pi * epoch / epochs)) / 2
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(optimizers[-1].param_groups[0]["lr"], stage["min_lr"], places=12)

    def test_constant_fallback_and_invalid_settings(self):
        optimizer = make_optimizer([torch.nn.Parameter(torch.ones(1))], 1e-4, 0.0)
        self.assertIsNone(make_scheduler(optimizer, {"epochs": 30}))
        self.assertIsNone(make_scheduler(optimizer, {"lr_scheduler": "constant"}))
        for cfg in ({"lr_scheduler": "unknown"},
                    {"lr_scheduler": "cosine", "epochs": 0},
                    {"lr_scheduler": "cosine", "epochs": 30, "min_lr": 1.0}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                make_scheduler(optimizer, cfg)


if __name__ == "__main__":
    unittest.main()
