import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def info(message):
    print(message)


def gaussian(window_size, sigma):
    values = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    return values / values.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(
        self,
        value_range=1.0,
        window_size=11,
        size_average=True,
    ):
        super(SSIM, self).__init__()
        self.value_range = value_range
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(self.window_size, self.channel)
        self.eps = 0.00001
        info(f"Init SSIM:")
        info(f"  value_range    : {value_range}")
        info(f"  window_size    : {window_size}")
        info(f"  size_average   : {size_average}")

    def forward(self, img1, img2):
        if img1.dim() == 5:
            img1 = torch.flatten(img1, start_dim=0, end_dim=1)
            img2 = torch.flatten(img2, start_dim=0, end_dim=1)

        img1 = img1 / self.value_range
        img2 = img2 / self.value_range
        (_, channel, _, _) = img1.size()
        if self.channel != channel:
            self.channel = channel
            self.window = create_window(self.window_size, self.channel)
        return _ssim(
            img1,
            img2,
            self.window,
            self.window_size,
            channel,
            self.size_average,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def average_logs(logs):
    out: Dict[str, float] = {}
    if not logs:
        return out
    for item in logs:
        for key, value in item.items():
            out[key] = out.get(key, 0.0) + float(value)
    return {key: value / len(logs) for key, value in out.items()}


def ensure_dir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str, model, extra=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "Checkpoint loaded non-strictly: "
            f"missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )
    return ckpt
