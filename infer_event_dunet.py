import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from adaptpro.training import amp_autocast, build_loaders, build_model, move_batch, read_config, resolve_device
from adaptpro.utils import SSIM, load_checkpoint, psnr
from adaptpro.visualization import Visualization


class ReferenceInferenceDiagnostics:
    """Observe affine reference injection without changing model predictions."""

    metric_names = (
        "gate", "contrast_abs", "shift_abs", "raw_correction_rms",
        "injected_rms", "injected_to_backbone_rms",
    )

    def __init__(self, model):
        branch = getattr(model, "reference_branch", None)
        if branch is None or branch.fusion_mode != "reference_affine":
            raise ValueError(
                "Reference diagnostics require an enabled reference_affine branch."
            )
        self.branch = branch
        self.gate = None
        self.affine = None
        self.sums = None
        self.valid_count = None
        self.seen_count = 0
        self.handles = []

    def __enter__(self):
        self.handles = [
            self.branch.gate.register_forward_hook(self._capture_gate),
            self.branch.fusion_out.register_forward_hook(self._capture_affine),
            self.branch.register_forward_hook(self._capture_injection, with_kwargs=True),
        ]
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.gate = None
        self.affine = None

    def _capture_gate(self, _module, _inputs, output):
        self.gate = torch.sigmoid(output.detach()).float()

    def _capture_affine(self, _module, _inputs, output):
        self.affine = torch.tanh(output.detach()).float().chunk(2, dim=1)

    def _capture_injection(self, _module, args, kwargs, output):
        if self.gate is None or self.affine is None:
            raise RuntimeError("Reference diagnostic hooks did not observe gate and affine outputs.")
        backbone = kwargs.get("backbone_feature", args[0] if args else None)
        reference_image = kwargs.get("reference_image", args[2] if len(args) > 2 else None)
        reference_valid = kwargs.get("reference_valid", args[3] if len(args) > 3 else None)
        if backbone is None or reference_image is None:
            raise RuntimeError("Reference diagnostic hooks could not read injector inputs.")

        valid = self.branch._validity_mask(reference_valid, reference_image).flatten().bool()
        self.seen_count += int(backbone.shape[0])
        if self.valid_count is None:
            self.valid_count = torch.zeros((), device=backbone.device, dtype=torch.float64)
            self.sums = torch.zeros(len(self.metric_names), device=backbone.device, dtype=torch.float64)
        self.valid_count += valid.sum()
        feature = backbone.detach().float()
        mean, std = self.branch._channel_statistics(feature, self.branch.statistics_eps)
        contrast, shift = self.affine
        raw_correction = contrast * (feature - mean) + shift * std
        injected = output.detach().float() - feature
        raw_rms = raw_correction.square().mean(dim=(1, 2, 3)).sqrt()
        injected_rms = injected.square().mean(dim=(1, 2, 3)).sqrt()
        backbone_rms = feature.square().mean(dim=(1, 2, 3)).sqrt()
        per_frame = torch.stack((
            self.gate.mean(dim=(1, 2, 3)),
            contrast.abs().mean(dim=(1, 2, 3)),
            shift.abs().mean(dim=(1, 2, 3)),
            raw_rms,
            injected_rms,
            injected_rms / backbone_rms.clamp_min(self.branch.statistics_eps),
        ), dim=1)
        self.sums += (per_frame.double() * valid[:, None]).sum(dim=0)
        self.gate = None
        self.affine = None

    def summary(self):
        valid_count = int(self.valid_count.item()) if self.valid_count is not None else 0
        if valid_count == 0:
            return None
        averages = (self.sums / valid_count).cpu().tolist()
        return dict(zip(self.metric_names, averages))

    def print_summary(self, label):
        summary = self.summary()
        if summary is None:
            print(f"[{label}] reference: valid=0/{self.seen_count}; no injection measured")
            return
        valid_count = int(self.valid_count.item())
        print(
            f"[{label}] reference: valid={valid_count}/{self.seen_count}, "
            f"scale={self.branch.residual_scale:.4f}, gate={summary['gate']:.4f}, "
            f"|contrast|={summary['contrast_abs']:.4f}, "
            f"|shift|={summary['shift_abs']:.4f}, "
            f"raw_correction_rms={summary['raw_correction_rms']:.6f}, "
            f"injected_rms={summary['injected_rms']:.6f}, "
            f"injected/backbone={summary['injected_to_backbone_rms']:.4%}"
        )


@torch.no_grad()
def run_inference(
    model, loader, device, output_dir, save_gt=False, amp_enabled=False,
    print_reference_diagnostics=False, reference_diagnostics_interval=100,
):
    if reference_diagnostics_interval < 0:
        raise ValueError("reference_diagnostics_interval must be non-negative.")
    model.eval()
    visualizer = Visualization(output_dir, save_keys=("gt",) if save_gt else ())
    ssim = SSIM().to(device)
    metric_logs = []
    diagnostics = ReferenceInferenceDiagnostics(model) if print_reference_diagnostics else None
    diagnostics_context = diagnostics if diagnostics is not None else nullcontext()

    with diagnostics_context:
        progress = tqdm(loader, desc="infer", leave=True)
        last_interval = 0
        for raw_batch in progress:
            batch = move_batch(raw_batch, device)
            with amp_autocast(device, amp_enabled):
                pred = model(batch)
            metric_pred = pred.float()

            batch_psnr = psnr(metric_pred, batch["gt"])
            batch_ssim = float(ssim(metric_pred, batch["gt"]).detach().cpu())
            metric_logs.append({"psnr": batch_psnr, "ssim": batch_ssim})
            avg_psnr = sum(item["psnr"] for item in metric_logs) / len(metric_logs)
            avg_ssim = sum(item["ssim"] for item in metric_logs) / len(metric_logs)
            progress.set_postfix(psnr=f"{avg_psnr:.2f}", ssim=f"{avg_ssim:.4f}")

            visualizer.save_batch(raw_batch, metric_pred)
            if diagnostics is not None and reference_diagnostics_interval:
                interval = diagnostics.seen_count // reference_diagnostics_interval
                if interval > last_interval:
                    diagnostics.print_summary(f"infer {diagnostics.seen_count}")
                    last_interval = interval

    if diagnostics is not None:
        diagnostics.print_summary("infer final")

    if metric_logs:
        avg_psnr = sum(item["psnr"] for item in metric_logs) / len(metric_logs)
        avg_ssim = sum(item["ssim"] for item in metric_logs) / len(metric_logs)
        print(
            f"Inference done: psnr={avg_psnr:.4f}, ssim={avg_ssim:.4f}, "
            f"output_dir={visualizer.output_dir}"
        )
    else:
        print(f"Inference done: no samples, output_dir={visualizer.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inference for Event-guided Mini-DUNet.")
    parser.add_argument("--config", default="configs/event_mini_dunet_infer.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["train", "validate", "query", "val", "test"], default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-gt", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    args = parser.parse_args()

    cfg = read_config(args.config)
    infer_cfg = cfg.get("inference", {})
    checkpoint = args.checkpoint or infer_cfg.get("checkpoint")
    split = args.split or infer_cfg.get("split", "test")
    output_dir = args.output_dir or infer_cfg.get("output_dir", "outputs/event_mini_dunet_infer")
    save_gt = args.save_gt or infer_cfg.get("save_gt", False)
    strict = args.strict or infer_cfg.get("strict", False)
    print_reference_diagnostics = bool(infer_cfg.get("print_reference_diagnostics", False))
    reference_diagnostics_interval = int(infer_cfg.get("reference_diagnostics_interval", 100))
    if not checkpoint:
        raise ValueError("Inference needs a checkpoint. Set inference.checkpoint or pass --checkpoint.")

    device = resolve_device(cfg)
    amp_enabled = bool(infer_cfg.get("amp", True)) and device.type == "cuda"
    include_query = split in {"validate", "query", "val"}
    train_loader, query_loader, test_loader = build_loaders(cfg, include_query=include_query)
    model = build_model(cfg, train_loader, device)
    load_checkpoint(checkpoint, model, strict=strict)

    loaders = {
        "train": train_loader,
        "validate": query_loader,
        "query": query_loader,
        "val": query_loader,
        "test": test_loader,
    }
    loader = loaders[split]
    run_inference(
        model,
        loader,
        device,
        Path(output_dir),
        save_gt=save_gt,
        amp_enabled=amp_enabled,
        print_reference_diagnostics=print_reference_diagnostics,
        reference_diagnostics_interval=reference_diagnostics_interval,
    )


if __name__ == "__main__":
    main()
