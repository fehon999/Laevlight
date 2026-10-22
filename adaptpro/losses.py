import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from adaptpro.utils import SSIM


class EnhancementLoss(nn.Module):
    """Paper-style Phi/Psi objectives and omega outer objective.

    Phi optimizes theta / SR on training data.
    Psi evaluates omega / DE on validation data.
    Omega combines Psi with the task-aware Phi difference from Eq. (5).
    """

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        lpips_weight: float = 0.5,
        charbonnier_eps: float = 1e-4,
        lpips_net: str = "alex",
        psi_charbonnier_weight: Optional[float] = None,
        psi_lpips_weight: Optional[float] = None,
        omega_gamma: float = 1.0,
        ssim_weight: float = 0.0,
        psi_ssim_weight: Optional[float] = None,
    ):
        super().__init__()
        if charbonnier_eps <= 0:
            raise ValueError(f"charbonnier_eps must be positive, got {charbonnier_eps}.")
        psi_ssim_weight = ssim_weight if psi_ssim_weight is None else psi_ssim_weight
        for name, weight in (("ssim_weight", ssim_weight), ("psi_ssim_weight", psi_ssim_weight)):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {weight}.")
        self.phi_weights = {
            "charbonnier": charbonnier_weight,
            "lpips": lpips_weight,
            "ssim": ssim_weight,
        }
        self.psi_weights = {
            "charbonnier": (
                charbonnier_weight
                if psi_charbonnier_weight is None
                else psi_charbonnier_weight
            ),
            "lpips": lpips_weight if psi_lpips_weight is None else psi_lpips_weight,
            "ssim": psi_ssim_weight,
        }
        self.charbonnier_eps = charbonnier_eps
        self.omega_gamma = omega_gamma
        needs_lpips = self.phi_weights["lpips"] > 0 or self.psi_weights["lpips"] > 0
        self.lpips = self._build_lpips(lpips_net) if needs_lpips else None
        self.ssim = SSIM() if ssim_weight > 0 or psi_ssim_weight > 0 else None

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
        # event: Optional[torch.Tensor] = None,
        # degradation_maps: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        terms = self.reconstruction_terms(pred, target)
        loss, logs = self.combine_terms(terms, self.phi_weights)
        logs = {f"phi_{key}": value for key, value in logs.items()}
        logs["total"] = logs["phi_total"]
        return loss, logs

    def psi_forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        # event: Optional[torch.Tensor] = None,
        # degradation_maps: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        terms = self.reconstruction_terms(pred, target)
        loss, logs = self.combine_terms(terms, self.psi_weights)
        logs = {f"psi_{key}": value for key, value in logs.items()}
        logs["total"] = logs["psi_total"]
        return loss, logs

    def omega_forward(
        self,
        psi_loss: torch.Tensor,
        phi_tilde_loss: torch.Tensor,
        phi_hat_loss: torch.Tensor,
        psi_logs: Optional[Dict[str, float]] = None,
        phi_tilde_logs: Optional[Dict[str, float]] = None,
        phi_hat_logs: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        penalty = phi_tilde_loss - phi_hat_loss
        loss = psi_loss + self.omega_gamma * penalty

        logs = {}
        if psi_logs:
            logs.update({f"outer_{key}": value for key, value in psi_logs.items() if key != "total"})
        if phi_tilde_logs:
            logs.update({f"tilde_{key}": value for key, value in phi_tilde_logs.items() if key != "total"})
        if phi_hat_logs:
            logs.update({f"hat_{key}": value for key, value in phi_hat_logs.items() if key != "total"})
        logs["omega_psi"] = float(psi_loss.detach().cpu())
        logs["omega_phi_tilde"] = float(phi_tilde_loss.detach().cpu())
        logs["omega_phi_hat"] = float(phi_hat_loss.detach().cpu())
        logs["omega_penalty"] = float(penalty.detach().cpu())
        logs["total"] = float(loss.detach().cpu())
        return loss, logs

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,

    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.phi_forward(pred, target)
