import math
from typing import Dict, Tuple

import torch
from torch import nn

from adaptpro.utils import SSIM


class EnhancementLoss(nn.Module):
    """Original joint-pretraining reconstruction objective."""

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        lpips_weight: float = 0.5,
        charbonnier_eps: float = 1e-4,
        lpips_net: str = "alex",
        ssim_weight: float = 0.0,
    ):
        super().__init__()
        if charbonnier_eps <= 0:
            raise ValueError(f"charbonnier_eps must be positive, got {charbonnier_eps}.")
        if not math.isfinite(ssim_weight) or ssim_weight < 0:
            raise ValueError(f"ssim_weight must be finite and non-negative, got {ssim_weight}.")
        self.phi_weights = {
            "charbonnier": charbonnier_weight,
            "lpips": lpips_weight,
            "ssim": ssim_weight,
        }
        self.charbonnier_eps = charbonnier_eps
        needs_lpips = self.phi_weights["lpips"] > 0
        self.lpips = self._build_lpips(lpips_net) if needs_lpips else None
        self.ssim = SSIM() if ssim_weight > 0 else None

    @staticmethod
    def _build_lpips(lpips_net: str):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS loss needs the `lpips` package. Install it with `pip install lpips`."
            ) from exc

        model = lpips.LPIPS(net=lpips_net).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_squared = (pred - target).square().mean(dim=1, keepdim=True)
        return torch.sqrt(diff_squared + self.charbonnier_eps).mean()

    def lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips is None:
            return pred.new_tensor(0.0)
        return self.lpips(pred, target, normalize=True).mean()

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ssim is None:
            return pred.new_tensor(0.0)
        # Keep the same RGB/window/range convention as evaluation. FP32 and
        # disabled autocast are important for local variances and gradients.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return 1.0 - self.ssim(pred.float(), target.float())

    def reconstruction_terms(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        terms = {
            "charbonnier": self.charbonnier_loss(pred, target),
            "lpips": self.lpips_loss(pred, target),
        }
        if self.ssim is not None:
            terms["ssim"] = self.ssim_loss(pred, target)
        return terms

    @staticmethod
    def combine_terms(terms: Dict[str, torch.Tensor], weights: Dict[str, float]):
        total = next(iter(terms.values())).new_tensor(0.0)
        logs = {}
        for name, value in terms.items():
            total = total + weights[name] * value
            logs[name] = float(value.detach().cpu())
        logs["total"] = float(total.detach().cpu())
        return total, logs

    def phi_forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        terms = self.reconstruction_terms(pred, target)
        loss, logs = self.combine_terms(terms, self.phi_weights)
        logs = {f"phi_{key}": value for key, value in logs.items()}
        logs["total"] = logs["phi_total"]
        return loss, logs

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,

    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.phi_forward(pred, target)
