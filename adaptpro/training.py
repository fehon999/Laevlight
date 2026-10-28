import inspect
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from adaptpro.models import get_model
from adaptpro.data import get_dataset, get_split_dataset
from adaptpro.losses import EnhancementLoss
from adaptpro.utils import SSIM, average_logs, ensure_dir, psnr, save_checkpoint, seed_everything


def read_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{requested}` was requested, but CUDA is not available.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device `{requested}` was requested, but only {torch.cuda.device_count()} device(s) are visible."
        )
    return device


def amp_autocast(device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device, enabled=True):
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def build_loaders(cfg):
    """Load only the training and test splits used by the original pretraining."""
    train_set, test_set = get_dataset(cfg.get("data"))
    loader_cfg = cfg.get("loader", {})
    train_loader = DataLoader(
        train_set,
        batch_size=loader_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=loader_cfg.get("drop_last", True),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=loader_cfg.get("test_batch_size", 1),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, test_loader


def build_eval_loader(cfg, split="test"):
    """Inference needs only the selected split, including for channel checks."""
    data_cfg = cfg.get("data") or {}
    split = data_cfg.get(f"{split}_split", split)
    dataset = get_split_dataset(data_cfg, split=split, is_train=False)
    loader_cfg = cfg.get("loader", {})
    return DataLoader(
        dataset,
        batch_size=loader_cfg.get("test_batch_size", 1),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
    )


def infer_event_channels(cfg, loader):
    configured = cfg["model"].get("event_channels")
    try:
        sample = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Data loader is empty. For training, reduce batch_size or set loader.drop_last=false.") from exc
    if "event" not in sample or sample["event"].ndim != 4:
        shape = None if "event" not in sample else tuple(sample["event"].shape)
        raise ValueError(f"Dataset samples must contain `event` with shape [B, C, H, W], got {shape}.")
    actual = int(sample["event"].shape[1])
    if configured is not None and int(configured) != actual:
        raise ValueError(
            f"model.event_channels={configured} does not match the dataset event channels ({actual})."
        )
    return actual


def build_model(cfg, loader, device):
    event_channels = infer_event_channels(cfg, loader)
    model = get_model(cfg, event_channels)

    return model.to(device)


def build_criterion(cfg, device):
    loss_cfg = dict(cfg.get("loss", {}))
    supported_keys = set(inspect.signature(EnhancementLoss.__init__).parameters) - {"self"}
    unsupported_keys = set(loss_cfg) - supported_keys
    if unsupported_keys:
        raise ValueError(f"Unsupported loss config keys: {sorted(unsupported_keys)}")
    return EnhancementLoss(**loss_cfg).to(device)


def prepare_runtime(cfg):
    seed_everything(cfg.get("seed", 42))
    device = resolve_device(cfg)
    train_loader, test_loader = build_loaders(cfg)
    model = build_model(cfg, train_loader, device)
    criterion = build_criterion(cfg, device)
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/event_mini_dunet"))
    return device, train_loader, test_loader, model, criterion, output_dir


def move_batch(batch, device):
    moved = {
        "low": batch["low"].float().to(device, non_blocking=True),
        "event": batch["event"].float().to(device, non_blocking=True),
        "gt": batch["gt"].float().to(device, non_blocking=True),
    }
    if "lowlight_image_blur" in batch:
        moved["lowlight_image_blur"] = batch["lowlight_image_blur"].float().to(device, non_blocking=True)
    if "img_refer" in batch:
        moved["img_refer"] = batch["img_refer"].float().to(device, non_blocking=True)
    if "reference_valid" in batch:
        moved["reference_valid"] = batch["reference_valid"].to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
    if "ill_list" in batch:
        moved["ill_list"] = [
            item.float().to(device, non_blocking=True) if torch.is_tensor(item) else item
            for item in batch["ill_list"]
        ]
    return moved


def run_forward(model, criterion, batch):
    pred = model(batch)
    loss, logs = criterion(pred, batch["gt"])
    return loss, logs, pred


def update_progress_bar(progress, logs):
    avg = average_logs(logs)
    progress.set_postfix(
        loss=f"{avg.get('total', 0.0):.4f}",
        psnr=f"{avg.get('psnr', 0.0):.2f}",
        ssim=f"{avg.get('ssim_metric', 0.0):.4f}",
    )


def train_standard_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    desc,
    scaler=None,
    amp_enabled=False,
):
    model.train()
    logs = []
    ssim=SSIM()
    progress = tqdm(loader, desc=desc, leave=False)
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, amp_enabled):
            loss, batch_logs, pred = run_forward(model, criterion, batch)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            metric_pred = pred.float()
            batch_logs["psnr"] = psnr(metric_pred, batch["gt"])
            batch_logs["ssim_metric"] = float(ssim(metric_pred, batch["gt"]).detach().cpu())
        batch_logs["total"] = float(loss.detach().cpu())
        logs.append(batch_logs)
        update_progress_bar(progress, logs)
    return average_logs(logs)


@torch.no_grad()
def validate(model, loader, criterion, device, desc="val", amp_enabled=False):
    model.eval()
    logs = []
    ssim=SSIM()
    progress = tqdm(loader, desc=desc, leave=False)
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        with amp_autocast(device, amp_enabled):
            loss, batch_logs, pred = run_forward(model, criterion, batch)
        metric_pred = pred.float()
        batch_logs["psnr"] = psnr(metric_pred, batch["gt"])
        batch_logs["ssim_metric"] = float(ssim(metric_pred, batch["gt"]).detach().cpu())
        batch_logs["total"] = float(loss.detach().cpu())
        logs.append(batch_logs)
        update_progress_bar(progress, logs)
    return average_logs(logs)


def make_optimizer(params, lr, weight_decay):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, train_cfg):
    """Build the original epoch-based pretraining schedule."""
    name = train_cfg.get("lr_scheduler", "constant")
    if name == "constant":
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {name!r}")
    epochs = train_cfg.get("epochs", 0)
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    if epochs < 1:
        raise ValueError("Cosine scheduling requires epochs >= 1.")
    if not 0.0 <= min_lr <= min(group["lr"] for group in optimizer.param_groups):
        raise ValueError("min_lr must be between zero and each initial learning rate.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr,
    )


def log_line(stage, epoch, train_logs, eval_logs, eval_name="test"):
    train_msg = ", ".join(
        f"{k}={v:.4f}"
        for k, v in train_logs.items()
        if k in {"total", "psnr", "ssim_metric"}
    )
    eval_msg = ", ".join(f"{k}={v:.4f}" for k, v in eval_logs.items() if k in {"total", "psnr", "ssim_metric"})
    print(f"[{stage}] epoch {epoch}: train({train_msg}) {eval_name}({eval_msg})")


def save_progress(output_dir, model, cfg, stage, epoch, eval_logs, best_psnr, eval_name="test"):
    save_checkpoint(
        Path(output_dir) / f"{stage}_latest.pt",
        model,
        {"config": cfg, "stage": stage, "epoch": epoch, eval_name: eval_logs},
    )
    current_psnr = eval_logs.get("psnr", -1.0)
    if current_psnr > best_psnr:
        save_checkpoint(
            Path(output_dir) / "best.pt",
            model,
            {"config": cfg, "stage": stage, "epoch": epoch, "best_psnr": current_psnr},
        )
        return current_psnr
    return best_psnr
