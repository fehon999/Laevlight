import argparse
from pathlib import Path

from adaptpro.training import (
    log_line,
    make_grad_scaler,
    make_optimizer,
    prepare_runtime,
    read_config,
    save_progress,
    set_trainable,
    train_standard_epoch,
    validate_parameter_partition,
    validate,
)
from adaptpro.utils import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Task adaptation with a frozen reference feature meta encoder."
    )
    parser.add_argument("--config", default="configs/event_mini_dunet_finetune.yaml")
    parser.add_argument("--checkpoint", default=None, help="Global-training checkpoint used to initialize fine-tuning.")
    parser.add_argument("--resume", default=None, help="Optional fine-tuning checkpoint to continue from.")
    parser.add_argument("--allow-random-init", action="store_true", help="Allow fine-tuning without a checkpoint.")
    args = parser.parse_args()

    cfg = read_config(args.config)
    device, train_loader, _query_loader, test_loader, model, criterion, output_dir = prepare_runtime(
        cfg,
        include_query=False,
    )

    finetune_cfg = cfg.get("finetune", {})
    init_checkpoint = args.checkpoint or finetune_cfg.get("init_checkpoint")
    if args.resume:
        load_checkpoint(args.resume, model, strict=True)
    elif init_checkpoint:
        # A baseline EvLight checkpoint has no reference-branch keys.  Load it
        # non-strictly by default, while keeping resumed adapter runs strict.
        load_checkpoint(
            init_checkpoint,
            model,
            strict=bool(finetune_cfg.get("init_strict", False)),
        )
        if finetune_cfg.get("reset_recovery", True):
            model.reset_recovery_parameters()
            print("Reset task theta; kept the reference feature encoder omega frozen.")
    elif not args.allow_random_init:
        raise ValueError("Fine-tuning needs --checkpoint or finetune.init_checkpoint.")

    train_cfg = cfg.get("train", {})
    adapt = train_cfg.get("adaptation", {})
    weight_decay = train_cfg.get("weight_decay", 3e-4)
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)
    best_psnr = -1.0
    validate_parameter_partition(model)

    set_trainable(model.shared_parameters(), False)
    set_trainable(model.meta_parameters(), False)
    set_trainable(model.recovery_parameters(), True)

    if adapt.get("epochs", 0) > 0:
        optimizer = make_optimizer(model.recovery_parameters(), adapt.get("lr", 1e-4), weight_decay)
        stage_name = "adapt_evlight_frozen_reference_meta"
        for epoch in range(1, adapt.get("epochs", 0) + 1):
            train_logs = train_standard_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                stage_name,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            test_logs = validate(
                model,
                test_loader,
                criterion,
                device,
                desc="test",
                amp_enabled=amp_enabled,
            )
            log_line(stage_name, epoch, train_logs, test_logs, eval_name="test")
            best_psnr = save_progress(
                output_dir,
                model,
                cfg,
                stage_name,
                epoch,
                test_logs,
                best_psnr,
            )

    final_logs = validate(
        model,
        test_loader,
        criterion,
        device,
        desc="test",
        amp_enabled=amp_enabled,
    )
    save_checkpoint(Path(output_dir) / "finetune_last.pt", model, {"config": cfg, "test": final_logs})
    print(f"Saved fine-tuning checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
