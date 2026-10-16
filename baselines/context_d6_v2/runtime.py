"""Training runtime for the named V2 multiscale-attentive context experiment."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.B5.checkpoint import atomic_save, load_checkpoint, pack_rng, scales_digest, scales_payload
from baselines.B5.config import ModelConfig
from baselines.B5.runtime import device_from_name, seed_all
from baselines.context_d6.config import resolve_path
from baselines.context_d6.data import MachineD6Dataset
from baselines.context_d6.flow import flow_loss, sample
from baselines.context_d6.runtime import _loader, _preprocessor, _sha256, _tensor_batch
from ecg12gen.evaluate import _evaluate_record_subset, rmse_bonus, write_report

from . import ARCHITECTURE_ID, CONDITION_SCHEMA, CONTEXT_SOURCE, CONTEXT_VARIANT
from .model import ContextD6V2UNet, load_b5_weights
from baselines.context_d6_v3 import ContextD6V3UNet, load_b5_weights as load_b5_weights_v3
from baselines.context_d6_v4 import ContextD6V4UNet, load_b5_weights as load_b5_weights_v4
from baselines.context_d6_v5 import ContextD6V5UNet, load_b5_weights as load_b5_weights_v5
from baselines.context_d6_v6 import ContextD6V6UNet, load_b5_weights as load_b5_weights_v6


def _model_class(config: dict[str, Any]):
    version = str(config["context"].get("encoder_version", "v2_msconv_attnpool"))
    if version == "v2_msconv_attnpool":
        return ContextD6V2UNet, load_b5_weights, "B5-U-context-d6-v2-msconv-attnpool", "b5-demographics-plus-machine-record-context-v2"
    if version == "v3_anchor_query_pool":
        return ContextD6V3UNet, load_b5_weights_v3, "B5-U-context-d6-v3-anchor-query", "b5-demographics-plus-machine-record-context-v3"
    if version == "v4_dual_pool":
        return ContextD6V4UNet, load_b5_weights_v4, "B5-U-context-d6-v4-dual-pool", "b5-demographics-plus-machine-record-context-v4"
    if version == "v5_bottleneck_crossattn":
        return ContextD6V5UNet, load_b5_weights_v5, "B5-U-context-d6-v5-bottleneck-crossattn", "b5-demographics-plus-machine-record-context-v5"
    if version == "v6_reliability_gate":
        return ContextD6V6UNet, load_b5_weights_v6, "B5-U-context-d6-v6-reliability-gated", "b5-demographics-plus-machine-record-context-v6"
    raise ValueError(f"Unknown context encoder_version: {version}")


@torch.no_grad()
def validate(model: ContextD6V2UNet, dataset: MachineD6Dataset, config: dict[str, Any],
             device: torch.device, output_dir: Path) -> dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(dataset), 12, 5000)
    prediction = np.lib.format.open_memmap(output_dir / "prediction_uV.npy", mode="w+",
                                           dtype=np.float32, shape=shape)
    target = np.lib.format.open_memmap(output_dir / "target_uV.npy", mode="w+",
                                       dtype=np.float32, shape=shape)
    scale = dataset.preprocessor.scale_uV_by_source["d12"][None, :, None]
    settings, rows, offset = config["sampling"], [], 0
    import time
    started = time.perf_counter()
    for batch in _loader(dataset, config, False):
        inputs = _tensor_batch(batch, device)
        generated = sample(model, inputs, batch["keys"], int(settings["seed"]),
                           int(settings["steps"]), settings["solver"], int(settings["samples"]))
        size = len(generated)
        prediction[offset:offset + size] = generated.float().cpu().numpy() * scale
        target[offset:offset + size] = batch["target_uV"]
        rows.extend(batch["evaluation_metadata"])
        offset += size
    prediction.flush(); target.flush()
    summary, details = _evaluate_record_subset(prediction, target, "task2", rows)
    summary.update({
        "input_type": CONTEXT_SOURCE, "body_scale_variant": CONTEXT_VARIANT,
        "context_used": bool(config["context"]["enabled"]),
        "ecg_machine_d6_r_missing11": summary["r_missing11"],
        "ecg_machine_d6_task2_missing_lead_mean_rmse_uV": summary["task2_missing_lead_mean_rmse_uV"],
        "ecg_machine_d6_rmse_bonus_score": rmse_bonus(summary["task2_missing_lead_mean_rmse_uV"]),
        "architecture_id": _model_class(config)[2], "condition_schema": _model_class(config)[3],
        "solver": settings["solver"], "integration_steps": int(settings["steps"]),
        "samples": int(settings["samples"]), "seed": int(settings["seed"]),
        "elapsed_seconds": time.perf_counter() - started,
    })
    write_report(output_dir, summary, [{"input_type": CONTEXT_SOURCE, **row} for row in details],
                 title="V2 multiscale attentive machine-context experiment")
    with (output_dir / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return summary


def run_training(config: dict[str, Any], device_name: str) -> None:
    device = device_from_name(device_name)
    training = config["training"]
    seed_all(int(training["seed"]), bool(training["deterministic"]))
    base_path = resolve_path(config, "init_checkpoint")
    base = load_checkpoint(base_path)
    preprocessor = _preprocessor(config, base)
    model_config = ModelConfig.from_dict(config["model"])
    context_dim = int(config["context"]["dimension"])
    model_type, loader, architecture_id, condition_schema = _model_class(config)
    model_kwargs = {}
    if str(config["context"].get("encoder_version", "")) in {
            "v4_dual_pool", "v5_bottleneck_crossattn", "v6_reliability_gate"}:
        model_kwargs["initial_query_fraction"] = float(
            config["context"].get("initial_query_fraction", 0.2))
        model_kwargs["injection_layers"] = tuple(
            int(layer) for layer in config["context"].get("injection_layers", [0, 1, 2]))
    if str(config["context"].get("encoder_version", "")) == "v5_bottleneck_crossattn":
        model_kwargs["cross_attention_heads"] = int(
            config["context"].get("cross_attention_heads", 4))
        model_kwargs["cross_attention_dropout"] = float(
            config["context"].get("cross_attention_dropout", 0.1))
    if str(config["context"].get("encoder_version", "")) == "v6_reliability_gate":
        model_kwargs["cross_attention_heads"] = int(
            config["context"].get("cross_attention_heads", 4))
        model_kwargs["cross_attention_dropout"] = float(
            config["context"].get("cross_attention_dropout", 0.1))
        model_kwargs["reliability_hidden"] = int(
            config["context"].get("reliability_hidden", 64))
        model_kwargs["reliability_logit_clamp"] = float(
            config["context"].get("reliability_logit_clamp", 2.0))
    model = model_type(model_config, context_dim, **model_kwargs).to(device)
    loader(model, base["ema"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    amp = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    train_data = MachineD6Dataset(config, "train", preprocessor)
    validation_data = MachineD6Dataset(config, "validation", preprocessor)
    generator = torch.Generator().manual_seed(int(training["seed"]))
    loader = _loader(train_data, config, True, generator)
    output = resolve_path(config, "output_dir")
    if any((output / name).exists() for name in ("last.pt", "best.pt")):
        raise FileExistsError("Output already contains checkpoints; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    manifests = {
        "train": train_data.manifest_digest, "validation": validation_data.manifest_digest,
        "init_checkpoint": _sha256(base_path), "base_scales": _sha256(resolve_path(config, "scales")),
        "context_scales": _sha256(resolve_path(config, "context_scales")),
    }
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scale_tensor = torch.as_tensor(preprocessor.scale_uV_by_source["d12"], device=device)
    updates, best = 0, -float("inf")

    def payload(epoch: int) -> dict[str, Any]:
        return {
            "format_version": 1, "architecture_id": architecture_id,
            "condition_schema": condition_schema, "model_config": asdict(model_config),
            "context_config": dict(config["context"]), "context_source": CONTEXT_SOURCE,
            "context_variant": CONTEXT_VARIANT, "context_enabled": bool(config["context"]["enabled"]),
            "epoch": epoch, "updates": updates, "best_r_missing11": best,
            "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(), "rng": pack_rng(), "loader_rng": generator.get_state(),
            "scales": scales_payload(preprocessor), "scales_sha256": scales_digest(scales_payload(preprocessor)),
            "manifests": manifests, "training_config": dict(training), "loss_config": dict(config["loss"]),
            "sampling_config": dict(config["sampling"]), "base_checkpoint": str(base_path),
        }

    history = output / "history.jsonl"
    for epoch in range(int(training["epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}; count = 0
        for index, batch in enumerate(loader):
            inputs = _tensor_batch(batch, device)
            accumulation = int(training["gradient_accumulation"])
            group_start = (index // accumulation) * accumulation
            divisor = min(accumulation, len(loader) - group_start)
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                loss, parts = flow_loss(model, inputs, config["loss"], scale_tensor)
            scaler.scale(loss / divisor).backward()
            for key, value in {"loss": loss, **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(inputs["anchor"])
            count += len(inputs["anchor"])
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]),
                                               error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                decay = float(training["ema_decay"])
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        ema[key].mul_(decay).add_(value, alpha=1 - decay)
        record: dict[str, Any] = {"epoch": epoch + 1, "updates": updates,
                                  "train": {key: value / count for key, value in totals.items()}}
        if (epoch + 1) % int(training["validate_every"]) == 0 or epoch + 1 == int(training["epochs"]):
            evaluation = model_type(model_config, context_dim, **model_kwargs).to(device)
            evaluation.load_state_dict(ema, strict=True)
            summary = validate(evaluation, validation_data, config, device,
                               output / "validation" / f"epoch_{epoch + 1:04d}")
            score = float(summary["r_missing11"])
            record["validation"] = summary
            if score > best:
                best = score
                atomic_save(output / "best.pt", payload(epoch))
        atomic_save(output / "last.pt", payload(epoch))
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"epoch": epoch + 1, "updates": updates,
                          "loss": record["train"]["loss"], "best_r_missing11": best},
                         ensure_ascii=False), flush=True)

