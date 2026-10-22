import inspect
from contextlib import nullcontext
from pathlib import Path
from adaptpro.visualization import Visualization
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from adaptpro.models import get_model
from adaptpro.data import get_dataset
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


def stage_data_config(cfg, stage=None):
    data_cfg = dict(cfg.get("data") or {})
    root = data_cfg.get("root")
    if stage in {"pretrain", "pretrain_all"}:
        root = data_cfg.get("root_pre", data_cfg.get("pretrain_root", root))
    elif stage in {
        "bilevel",
        "bilevel_reference_outer",
        "bilevel_de_outer",
        "meta",
        "meta_de",
    }:
        root = data_cfg.get("root_meta", data_cfg.get("meta_root", root))
    elif root is None:
        root = data_cfg.get("root_meta", data_cfg.get("meta_root", data_cfg.get("root_pre", data_cfg.get("pretrain_root"))))
    if root is not None:
        data_cfg["root"] = root
    return data_cfg


def build_loaders(cfg, data_cfg=None, include_query=True):
    train_set, query_set, test_set = get_dataset(data_cfg or stage_data_config(cfg), include_query=include_query)
    loader_cfg = cfg.get("loader", {})
    train_loader = DataLoader(
        train_set,
        batch_size=loader_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=loader_cfg.get("drop_last", True),
    )
    query_loader = None
    if query_set is not None:
        query_loader = DataLoader(
            query_set,
            batch_size=loader_cfg.get("eval_batch_size", 1),
            shuffle=False,
            num_workers=loader_cfg.get("num_workers", 4),
            pin_memory=True,
        )
    test_loader = DataLoader(
        test_set,
        batch_size=loader_cfg.get("test_batch_size", 1),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, query_loader, test_loader


def build_stage_loaders(cfg, stage, include_query=True):
    return build_loaders(cfg, data_cfg=stage_data_config(cfg, stage), include_query=include_query)


def infer_event_channels(cfg, train_loader):
    configured = cfg["model"].get("event_channels")
    try:
        sample = next(iter(train_loader))
    except StopIteration as exc:
        raise ValueError("Training loader is empty. Reduce batch_size or set loader.drop_last=false.") from exc
    if "event" not in sample or sample["event"].ndim != 4:
        shape = None if "event" not in sample else tuple(sample["event"].shape)
        raise ValueError(f"Dataset samples must contain `event` with shape [B, C, H, W], got {shape}.")
    actual = int(sample["event"].shape[1])
    if configured is not None and int(configured) != actual:
        raise ValueError(
            f"model.event_channels={configured} does not match the dataset event channels ({actual})."
        )
    return actual


def build_model(cfg, train_loader, device):
    event_channels = infer_event_channels(cfg, train_loader)
    model=get_model(cfg,event_channels)

    return model.to(device)


def build_criterion(cfg, device):
    loss_cfg = dict(cfg.get("loss", {}))
    supported_keys = set(inspect.signature(EnhancementLoss.__init__).parameters) - {"self"}
    unsupported_keys = set(loss_cfg) - supported_keys
    if unsupported_keys:
        raise ValueError(f"Unsupported loss config keys: {sorted(unsupported_keys)}")
    return EnhancementLoss(**loss_cfg).to(device)


def prepare_runtime(cfg, data_cfg=None, include_query=True, stage=None):

    seed_everything(cfg.get("seed", 42))
    device = resolve_device(cfg)
    train_loader, query_loader, test_loader = build_loaders(
        cfg,
        data_cfg=data_cfg or stage_data_config(cfg, stage),
        include_query=include_query,
    )
    model = build_model(cfg, train_loader, device)
    criterion = build_criterion(cfg, device)
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/event_mini_dunet"))
    return device, train_loader, query_loader, test_loader, model, criterion, output_dir


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


def set_trainable(parameters, enabled: bool):
    for param in parameters:
        param.requires_grad_(enabled)


def validate_parameter_partition(model):
    """Validate that meta omega and task theta form disjoint model partitions."""

    groups = {
        "meta": list(model.meta_parameters()),
        "recovery": list(model.recovery_parameters()),
        "shared": list(model.shared_parameters()),
    }
    group_ids = {
        name: {id(parameter) for parameter in parameters}
        for name, parameters in groups.items()
    }
    names = tuple(group_ids)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = group_ids[left_name] & group_ids[right_name]
            if overlap:
                raise ValueError(
                    f"Parameter groups {left_name!r} and {right_name!r} overlap "
                    f"by {len(overlap)} parameter(s)."
                )

    assigned_ids = set().union(*group_ids.values())
    all_ids = {id(parameter) for parameter in model.parameters()}
    unassigned = all_ids - assigned_ids
    if unassigned:
        raise ValueError(
            f"Model has {len(unassigned)} parameter(s) outside meta/recovery/shared groups."
        )
    return {name: len(parameters) for name, parameters in groups.items()}


def run_forward(model, criterion, batch, loss_name="phi"):
    # pred, degradation_maps = model(batch["low"], batch["event"])
    pred= model(batch)
    if loss_name == "phi":
        loss, logs = criterion.phi_forward(pred, batch["gt"])
    elif loss_name == "psi":
        loss, logs = criterion.psi_forward(pred, batch["gt"])
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
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
            loss, batch_logs, pred = run_forward(model, criterion, batch, loss_name="phi")
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
            # batch_logs["ssim_metric"] = float((1.0 - ssim_loss(pred, batch["gt"])).detach().cpu())
        batch_logs["total"] = float(loss.detach().cpu())
        logs.append(batch_logs)
        update_progress_bar(progress, logs)
    return average_logs(logs)


def _first_optimizer_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _trainable_parameters(parameters):
    return [param for param in parameters if param.requires_grad]


def _snapshot_parameters(parameters):
    return [param.detach().clone() for param in parameters]


def _restore_parameters(parameters, values):
    with torch.no_grad():
        for param, value in zip(parameters, values):
            param.copy_(value)


def _infinite_loader(loader, name):
    """Repeat a loader without itertools.cycle caching every yielded batch."""

    if loader is None or len(loader) == 0:
        raise ValueError(f"{name} loader is empty.")
    while True:
        yield from loader


def simulate_recovery_step(
    model,
    criterion,
    batch,
    step_lr,
    scaler=None,
    amp_enabled=False,
):
    """Temporarily compute theta_hat = theta_tilde - beta * grad_theta Phi."""

    set_trainable(model.shared_parameters(), False)
    set_trainable(model.meta_parameters(), False)
    set_trainable(model.recovery_parameters(), True)

    recovery_params = _trainable_parameters(model.recovery_parameters())
    recovery_state = _snapshot_parameters(recovery_params)
    with amp_autocast(batch["low"].device, amp_enabled):
        phi_loss, _, _ = run_forward(model, criterion, batch, loss_name="phi")
    use_scaler = scaler is not None and scaler.is_enabled()
    grad_loss = scaler.scale(phi_loss) if use_scaler else phi_loss
    grads = torch.autograd.grad(grad_loss, recovery_params, allow_unused=True)
    inv_scale = 1.0 / scaler.get_scale() if use_scaler else 1.0
    with torch.no_grad():
        for param, grad in zip(recovery_params, grads):
            if grad is not None:
                param.add_(grad, alpha=-step_lr * inv_scale)
    return recovery_params, recovery_state


def train_bilevel_reference_outer_epoch(
    model,
    train_loader,
    query_loader,
    criterion,
    recovery_optimizer,
    meta_optimizer,
    device,
    inner_steps=1,
    scaler=None,
    amp_enabled=False,
):
    """Paper-style first-order bilevel training.

    Outer variable omega: reference-frame feature extractor only.
    Inner variable theta: EvLight plus current encoding/alignment/gated fusion.

    Each outer step first adapts theta with the current frozen omega, then
    freezes theta and updates omega once on a query batch.
    """

    model.train()
    logs = []
    ssim=SSIM()
    train_iterator = _infinite_loader(train_loader, "Training")
    query_iterator = _infinite_loader(query_loader, "Query")
    outer_steps = len(train_loader)
    progress = tqdm(range(outer_steps), desc="bilevel_reference_outer", leave=False)
    for _ in progress:
        # Inner loop: optimize the task-specific EvLight backbone while the
        # reference feature extractor stays fixed.
        set_trainable(model.shared_parameters(), False)
        set_trainable(model.meta_parameters(), False)
        set_trainable(model.recovery_parameters(), True)
        support_logs = None
        for _ in range(inner_steps):
            support = move_batch(next(train_iterator), device)
            recovery_optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, amp_enabled):
                loss, support_logs, _ = run_forward(model, criterion, support, loss_name="phi")
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(recovery_optimizer)
                scaler.update()
            else:
                loss.backward()
                recovery_optimizer.step()

        # Outer loop: optimize only the reference feature extractor with all
        # task modules fixed.
        # G = grad_w Psi(theta_tilde,w;D_query)
        #   + gamma * [grad_w Phi(theta_tilde,w;D_tr) - grad_w Phi(theta_hat,w;D_tr)]
        train_for_omega = next(train_iterator)
        query_for_omega = next(query_iterator)
        meta_optimizer.zero_grad(set_to_none=True)
        set_trainable(model.recovery_parameters(), False)
        set_trainable(model.meta_parameters(), True)

        query_batch = move_batch(query_for_omega, device)
        with amp_autocast(device, amp_enabled):
            psi_loss, psi_logs, _ = run_forward(model, criterion, query_batch, loss_name="psi")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(psi_loss).backward()
        else:
            psi_loss.backward()

        train_batch = move_batch(train_for_omega, device)
        with amp_autocast(device, amp_enabled):
            phi_tilde_loss, phi_tilde_logs, phi_tilde_pred = run_forward(
                model, criterion, train_batch, loss_name="phi"
            )
        #--------visualization
        # visualizer = Visualization("outputs/visualization")
        # visualizer.save_batch(phi_tilde_pred, pred)
        #____________
        weighted_phi_tilde = criterion.omega_gamma * phi_tilde_loss
        if scaler is not None and scaler.is_enabled():
            scaler.scale(weighted_phi_tilde).backward()
        else:
            weighted_phi_tilde.backward()

        step_lr = _first_optimizer_lr(recovery_optimizer)
        recovery_params, recovery_state = simulate_recovery_step(
            model,
            criterion,
            train_batch,
            step_lr,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        try:
            set_trainable(model.recovery_parameters(), False)
            set_trainable(model.meta_parameters(), True)
            with amp_autocast(device, amp_enabled):
                phi_hat_loss, phi_hat_logs, _ = run_forward(model, criterion, train_batch, loss_name="phi")
            weighted_phi_hat = -criterion.omega_gamma * phi_hat_loss
            if scaler is not None and scaler.is_enabled():
                scaler.scale(weighted_phi_hat).backward()
                scaler.step(meta_optimizer)
                scaler.update()
            else:
                weighted_phi_hat.backward()
                meta_optimizer.step()
        finally:
            _restore_parameters(recovery_params, recovery_state)

        _, query_logs = criterion.omega_forward(
            psi_loss.detach(),
            phi_tilde_loss.detach(),
            phi_hat_loss.detach(),
            psi_logs=psi_logs,
            phi_tilde_logs=phi_tilde_logs,
            phi_hat_logs=phi_hat_logs,
        )

        if support_logs is not None:
            query_logs["support_total"] = support_logs["total"]
        with torch.no_grad():
            metric_pred = phi_tilde_pred.float()
            query_logs["psnr"] = psnr(metric_pred, train_batch["gt"])
            query_logs["ssim_metric"] = float(ssim(metric_pred, train_batch["gt"]).detach().cpu())
        logs.append(query_logs)
        update_progress_bar(progress, logs)

    set_trainable(model.shared_parameters(), True)
    set_trainable(model.recovery_parameters(), True)
    set_trainable(model.meta_parameters(), True)
    return average_logs(logs)


def train_bilevel_de_outer_epoch(
    model,
    train_loader,
    query_loader,
    criterion,
    recovery_optimizer,
    de_optimizer,
    device,
    inner_steps=1,
    scaler=None,
    amp_enabled=False,
):
    """Backward-compatible alias; ``de_optimizer`` now optimizes reference omega."""

    return train_bilevel_reference_outer_epoch(
        model,
        train_loader,
        query_loader,
        criterion,
        recovery_optimizer,
        de_optimizer,
        device,
        inner_steps=inner_steps,
        scaler=scaler,
        amp_enabled=amp_enabled,
    )


def train_meta_de_epoch(
    model,
    loader,
    criterion,
    recovery_optimizer,
    de_optimizer,
    device,
    inner_steps=1,
    scaler=None,
    amp_enabled=False,
):
    """Backward-compatible alias for the old function name."""

    return train_bilevel_reference_outer_epoch(
        model,
        loader,
        loader,
        criterion,
        recovery_optimizer,
        de_optimizer,
        device,
        inner_steps=inner_steps,
        scaler=scaler,
        amp_enabled=amp_enabled,
    )


@torch.no_grad()
def validate(model, loader, criterion, device, desc="val", amp_enabled=False):
    model.eval()
    logs = []
    ssim=SSIM()
    progress = tqdm(loader, desc=desc, leave=False)
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        with amp_autocast(device, amp_enabled):
            loss, batch_logs, pred = run_forward(model, criterion, batch, loss_name="phi")
        metric_pred = pred.float()
        batch_logs["psnr"] = psnr(metric_pred, batch["gt"])
        batch_logs["ssim_metric"] = float(ssim(metric_pred, batch["gt"]).detach().cpu())
        batch_logs["total"] = float(loss.detach().cpu())
        logs.append(batch_logs)
        update_progress_bar(progress, logs)
        # --------visualization
        # visualizer = Visualization("outputs/visualization")
        # visualizer.save_batch(raw_batch, pred)
        # ____________
    return average_logs(logs)


def make_optimizer(params, lr, weight_decay):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, stage_cfg):
    """Build an opt-in, epoch-based cosine schedule for a training stage."""
    name = stage_cfg.get("lr_scheduler", "constant")
    if name == "constant":
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {name!r}")
    epochs = stage_cfg.get("epochs", 0)
    min_lr = float(stage_cfg.get("min_lr", 1e-6))
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
        if k in {"total", "support_total", "psnr", "ssim_metric"}
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
