import argparse
from pathlib import Path
from adaptpro.visualization import Visualization
from adaptpro.training import (
    build_stage_loaders,
    log_line,
    make_grad_scaler,
    make_optimizer,
    make_scheduler,
    prepare_runtime,
    read_config,
    save_progress,
    train_bilevel_reference_outer_epoch,
    train_standard_epoch,
    validate_parameter_partition,
    validate,
)
from adaptpro.utils import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Reference-meta / EvLight-task bilevel training."
    )
    parser.add_argument("--config", default="configs/event_mini_dunet_train.yaml")
    parser.add_argument("--resume", default=None, help="Optional checkpoint used to initialize global training.")

    # parser.add_argument("--resume", default="/home/user/dufeihan/adapt_self/outputs/event_mini_dunet_global/global_last.pt", help="Optional checkpoint used to initialize global training.")
    args = parser.parse_args()

    cfg = read_config(args.config)
    train_cfg = cfg.get("train", {})
    stages = train_cfg.get("stages", {})
    pretrain = stages.get("pretrain_all", {})
    bilevel = stages.get(
        "bilevel_reference_outer",
        stages.get("bilevel_de_outer", stages.get("meta_de", {})),
    )
    run_pretrain = pretrain.get("epochs", 0) > 0
    run_bilevel = bilevel.get("epochs", 0) > 0
    bilevel_stage = "bilevel_reference_outer"
    primary_stage = "pretrain_all" if run_pretrain else bilevel_stage

    device, primary_train_loader, primary_query_loader, primary_test_loader, model, criterion, output_dir = prepare_runtime(
        cfg,
        stage=primary_stage,
        include_query=primary_stage == bilevel_stage,
    )
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)
    partition = validate_parameter_partition(model)
    print(
        "Parameter partition: "
        f"reference_encoder_meta={partition['meta']}, "
        f"evlight_task={partition['recovery']}, "
        f"shared={partition['shared']}"
    )

    pretrain_train_loader = primary_train_loader if primary_stage == "pretrain_all" else None
    pretrain_test_loader = primary_test_loader if primary_stage == "pretrain_all" else None
    bilevel_train_loader = primary_train_loader if primary_stage == bilevel_stage else None
    bilevel_query_loader = primary_query_loader if primary_stage == bilevel_stage else None
    bilevel_test_loader = primary_test_loader if primary_stage == bilevel_stage else None

    if args.resume:
        print("Resuming from checkpoint...")
        load_checkpoint(
            args.resume,
            model,
            strict=bool(
                train_cfg.get(
                    "resume_strict",
                    not getattr(model, "reference_enabled", False),
                )
            ),
        )

    weight_decay = train_cfg.get("weight_decay", 3e-4)
    best_psnr = -1.0

    if run_pretrain:
        if pretrain_train_loader is None or pretrain_test_loader is None:
            pretrain_train_loader, _, pretrain_test_loader = build_stage_loaders(
                cfg,
                "pretrain_all",
                include_query=False,
            )
        optimizer = make_optimizer(model.parameters(), pretrain.get("lr", 1e-4), weight_decay)
        scheduler = make_scheduler(optimizer, pretrain)
        for epoch in range(1, pretrain.get("epochs", 0) + 1):
            print(f"[pretrain_all] epoch {epoch}: lr={optimizer.param_groups[0]['lr']:.8g}")
            train_logs = train_standard_epoch(
                model,
                pretrain_train_loader,
                criterion,
                optimizer,
                device,
                "pretrain_all",
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            test_logs = validate(
                model,
                pretrain_test_loader,
                criterion,
                device,
                desc="pretrain_test",
                amp_enabled=amp_enabled,
            )
            log_line("pretrain_all", epoch, train_logs, test_logs, eval_name="test")
            best_psnr = save_progress(output_dir, model, cfg, "pretrain_all", epoch, test_logs, best_psnr)
            if scheduler is not None:
                scheduler.step()

    if run_bilevel:
        best_psnr = -1.0
        if bilevel_train_loader is None or bilevel_query_loader is None or bilevel_test_loader is None:
            bilevel_train_loader, bilevel_query_loader, bilevel_test_loader = build_stage_loaders(
                cfg,
                bilevel_stage,
                include_query=True,
            )
        recovery_parameters = list(model.recovery_parameters())
        meta_parameters = list(model.meta_parameters())
        if not meta_parameters:
            raise ValueError(
                "bilevel_reference_outer requires model.ReferenceBranch.enabled=true "
                "so the reference feature extractor can serve as omega."
            )
        recovery_optimizer = make_optimizer(
            recovery_parameters,
            bilevel.get("lr_recovery", 1e-4),
            weight_decay,
        )
        meta_optimizer = make_optimizer(
            meta_parameters,
            bilevel.get("lr_meta", bilevel.get("lr_de", 5e-5)),
            weight_decay,
        )
        for epoch in range(1, bilevel.get("epochs", 0) + 1):
            train_logs = train_bilevel_reference_outer_epoch(
                model,
                bilevel_train_loader,
                bilevel_query_loader,
                criterion,
                recovery_optimizer,
                meta_optimizer,
                device,
                inner_steps=bilevel.get("inner_steps", 1),
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            test_logs = validate(
                model,
                bilevel_test_loader,
                criterion,
                device,
                desc="bilevel_test",
                amp_enabled=amp_enabled,
            )
            log_line(bilevel_stage, epoch, train_logs, test_logs, eval_name="test")
            best_psnr = save_progress(
                output_dir,
                model,
                cfg,
                bilevel_stage,
                epoch,
                test_logs,
                best_psnr,
            )

    final_test_loader = bilevel_test_loader or pretrain_test_loader or primary_test_loader
    final_logs = validate(
        model,
        final_test_loader,
        criterion,
        device,
        desc="test",
        amp_enabled=amp_enabled,
    )
    save_checkpoint(Path(output_dir) / "global_last.pt", model, {"config": cfg, "test": final_logs})
    print(f"Saved global-training checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
