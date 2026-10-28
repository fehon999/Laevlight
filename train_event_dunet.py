"""Complete, single-stage joint training with the original pretraining recipe."""

import argparse
from pathlib import Path

from adaptpro.training import (
    log_line,
    make_grad_scaler,
    make_optimizer,
    make_scheduler,
    prepare_runtime,
    read_config,
    save_progress,
    train_standard_epoch,
    validate,
)
from adaptpro.utils import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Single-stage EvLight joint training.")
    parser.add_argument("--config", required=True, help="Dataset config, e.g. configs/sde_in.yaml.")
    parser.add_argument(
        "--resume", default=None,
        help="Initialize model weights only; optimizer and schedule start from epoch 1.",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    train_cfg = cfg.get("train", {})
    if "stages" in train_cfg:
        raise ValueError("Use a dataset config with train.epochs and train.lr; staged training was removed.")
    epochs = train_cfg.get("epochs", 0)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("train.epochs must be a positive integer.")

    device, train_loader, test_loader, model, criterion, output_dir = prepare_runtime(cfg)
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)

    if args.resume:
        print("Initializing model weights from checkpoint...")
        load_checkpoint(
            args.resume,
            model,
            strict=bool(train_cfg.get("resume_strict", not getattr(model, "reference_enabled", False))),
        )

    optimizer = make_optimizer(
        model.parameters(), train_cfg.get("lr", 1e-4), train_cfg.get("weight_decay", 3e-4),
    )
    scheduler = make_scheduler(optimizer, train_cfg)
    best_psnr = -1.0
    # Preserve the original pretraining checkpoint names and metadata.
    stage = "pretrain_all"
    for epoch in range(1, epochs + 1):
        print(f"[{stage}] epoch {epoch}: lr={optimizer.param_groups[0]['lr']:.8g}")
        train_logs = train_standard_epoch(
            model, train_loader, criterion, optimizer, device, stage,
            scaler=scaler, amp_enabled=amp_enabled,
        )
        test_logs = validate(
            model, test_loader, criterion, device, desc="pretrain_test", amp_enabled=amp_enabled,
        )
        log_line(stage, epoch, train_logs, test_logs, eval_name="test")
        best_psnr = save_progress(output_dir, model, cfg, stage, epoch, test_logs, best_psnr)
        if scheduler is not None:
            scheduler.step()

    final_logs = validate(model, test_loader, criterion, device, desc="test", amp_enabled=amp_enabled)
    save_checkpoint(Path(output_dir) / "global_last.pt", model, {"config": cfg, "test": final_logs})
    print(f"Training complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
