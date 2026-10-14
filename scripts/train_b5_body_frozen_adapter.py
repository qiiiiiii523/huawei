"""Frozen-B5 body-context adapter pilot; keeps all previous runs unchanged."""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, checkpoint_preprocessor, load_checkpoint, pack_rng, restore_rng
from baselines.B5.config import ModelConfig
from baselines.B5.context_body import B5BodyContext
from baselines.B5.context_body_meanstd import B5BodyContextMeanStd
from baselines.B5.losses import flow_loss
from baselines.B5.metadata import DemographicsTable
from baselines.B5.objective import file_sha256
from baselines.B5.runtime import seed_all
from ecg12gen.preprocessing import PreprocessingConfig
from scripts.prepare_task2_context_scales import with_context_scales
from scripts.train_b5_body_context import BatchAdapter, BodySubset, LOSS, ROOT, on_device, validate

NAME = "A_frozen_adapter_v1"
MEANSTD_NAME = "A_frozen_meanstd_v1"
SOURCE = ROOT / "results/B5/context_start/best.pt"
OUTPUT = ROOT / "results/context_body" / NAME


def adapter_name(name: str) -> bool:
    return name.startswith(("body_encoder.", "body_film."))


def assert_frozen(model: B5BodyContext, source: dict) -> None:
    for name, value in source["ema"].items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), value):
            raise ValueError(f"Frozen B5 tensor changed: {name}")


def make_model(source: dict, device: torch.device, encoder_variant: str = "mean") -> B5BodyContext:
    if encoder_variant not in ("mean", "mean_std"):
        raise ValueError(f"Unknown encoder variant: {encoder_variant}")
    model_class = B5BodyContext if encoder_variant == "mean" else B5BodyContextMeanStd
    model = model_class(ModelConfig.from_dict(source["model_config"]))
    missing, unexpected = model.load_state_dict(source["ema"], strict=False)
    if unexpected or not missing or not all(adapter_name(name) for name in missing):
        raise ValueError("B5/adapter state mismatch")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(adapter_name(name))
    assert_frozen(model, source)
    return model.to(device).eval()


def run(args) -> None:
    if min(args.epochs, args.batch_size, args.accumulation, args.validate_every) < 1 or args.learning_rate <= 0:
        raise ValueError("Training settings must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    seed_all(42, deterministic=True)
    source = load_checkpoint(SOURCE)
    base = checkpoint_preprocessor(source, PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml"))
    preprocessor = with_context_scales(base, ROOT / "results/B5/context_params/body_A_scales.npz",
                                       "body_scale_d6", "A_raw_window")
    encoder_variant = getattr(args, "encoder_variant", "mean")
    experiment_name = NAME if encoder_variant == "mean" else MEANSTD_NAME
    output = ROOT / "results/context_body" / experiment_name
    model = make_model(source, device, encoder_variant)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or not any(p.requires_grad for p in model.body_encoder.parameters()) or not any(
            p.requires_grad for p in model.body_film.parameters()):
        raise ValueError("Both adapter modules must be trainable")
    train_data, val_data = BodySubset("train", "A_raw_window"), BodySubset("validation", "A_raw_window")
    try:
        demographics = DemographicsTable(ROOT / "data/userinfobean.csv")
        train_adapter = BatchAdapter(preprocessor, demographics, 0.1, seed=42)
        val_adapter = BatchAdapter(preprocessor, demographics, 0.0, seed=42)
        generator = torch.Generator().manual_seed(42)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, num_workers=0, collate_fn=train_adapter,
                                  pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=val_adapter, pin_memory=device.type == "cuda")
        scale = torch.as_tensor(preprocessor.scale_uV_by_source["d12"], device=device)
        # Zero-initialized FiLM should receive gradients; frozen B5 should not.
        rng_before, collate_before, loader_before = pack_rng(), train_adapter.rng.bit_generator.state, generator.get_state()
        batch = on_device(next(iter(train_loader)), device)
        loss, _ = flow_loss(model, batch, LOSS, scale, body_context=True)
        loss.backward()
        film_grad = sum(float(p.grad.detach().abs().sum()) for p in model.body_film.parameters() if p.grad is not None)
        if film_grad <= 0 or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("Adapter gradient/freeze preflight failed")
        model.zero_grad(set_to_none=True)
        restore_rng(rng_before)
        train_adapter.rng.bit_generator.state = collate_before
        generator.set_state(loader_before)
        print(json.dumps({"preflight": "passed", "film_gradient_l1": film_grad,
                          "trainable_parameters": sum(p.numel() for p in trainable)}), flush=True)
        if args.preflight:
            return
        settings = {"experiment": experiment_name, "source_checkpoint_sha256": file_sha256(SOURCE),
                    "variant": "A_raw_window", "train_windows": len(train_data),
                    "validation_windows": len(val_data), "epochs": args.epochs,
                    "batch_size": args.batch_size, "accumulation": args.accumulation,
                    "learning_rate": args.learning_rate, "validate_every": args.validate_every,
                    "steps": 16, "samples": 16, "seed": 42, "context_dropout": 0.1,
                    "loss": LOSS, "optimizer": "AdamW_adapter_only", "ema_decay": 0.999,
                    "frozen_backbone": True, "backbone_mode": "eval_with_gradients"}
        if encoder_variant == "mean_std":
            settings["encoder_variant"] = encoder_variant
        ledger = output / "run_config.json"
        if args.resume:
            if not ledger.is_file() or json.loads(ledger.read_text(encoding="utf-8")) != settings:
                raise ValueError("Resume configuration mismatch")
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Experiment output already exists: {output}")
            output.mkdir(parents=True, exist_ok=True)
            ledger.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        start_epoch, updates, best_r = 0, 0, float("-inf")
        if args.resume:
            saved = torch.load(output / "last.pt", map_location=device, weights_only=True)
            if saved.get("run_config") != settings:
                raise ValueError("Resume checkpoint mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            ema = saved["ema"]
            start_epoch, updates, best_r = saved["epoch"], saved["updates"], saved["best_r"]
            generator.set_state(saved["loader_rng"])
            train_adapter.rng.bit_generator.state = saved["collate_rng"]
            restore_rng(saved["rng"])
        assert_frozen(model, source)
        for name, value in source["ema"].items():
            if not torch.equal(ema[name].detach().cpu(), value):
                raise ValueError(f"EMA frozen B5 tensor differs: {name}")
        for epoch in range(start_epoch, args.epochs):
            model.eval()  # Keep frozen metadata dropout off; gradients still flow through B5.
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for index, raw in enumerate(train_loader):
                batch = on_device(raw, device)
                group_start = (index // args.accumulation) * args.accumulation
                divisor = min(args.accumulation, len(train_loader) - group_start)
                loss, _ = flow_loss(model, batch, LOSS, scale, body_context=True)
                (loss / divisor).backward()
                running += float(loss.detach())
                if (index + 1) % args.accumulation == 0 or index + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            if adapter_name(name):
                                ema[name].mul_(0.999).add_(value, alpha=0.001)
            assert_frozen(model, source)
            record = {"epoch": epoch + 1, "updates": updates,
                      "train_loss": running / len(train_loader)}
            improved = False
            if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
                evaluation = make_model(source, device, encoder_variant)
                evaluation.load_state_dict(ema, strict=True)
                summary = validate(evaluation, val_loader, preprocessor, experiment_name, device,
                                   output / "validation" / f"epoch_{epoch + 1:04d}", 16, 16)
                record["validation"] = summary
                improved = summary["r_missing11"] > best_r
                if improved:
                    best_r = summary["r_missing11"]
                del evaluation
            payload = {"format": "b5-body-frozen-adapter-v1", "run_config": settings,
                       "epoch": epoch + 1, "updates": updates, "best_r": best_r,
                       "model": model.state_dict(), "ema": ema,
                       "optimizer": optimizer.state_dict(), "rng": pack_rng(),
                       "loader_rng": generator.get_state(),
                       "collate_rng": train_adapter.rng.bit_generator.state}
            if improved:
                atomic_save(output / "best.pt", payload)
            atomic_save(output / "last.pt", payload)
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"epoch": epoch + 1, "updates": updates,
                              "loss": record["train_loss"], "best_body_r": best_r}), flush=True)
    finally:
        train_data.close()
        val_data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--encoder-variant", choices=("mean", "mean_std"), default="mean")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
