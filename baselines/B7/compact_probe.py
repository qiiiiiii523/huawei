"""Short synthetic training probe for compact B7-T GPU throughput and memory.

Runs the actual FP32 model, FM/MAE/PCC, AdamW, clipping and EMA. No dataset
preparation, evaluation, production checkpoint or baseline configuration is changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .compact import PRESETS, load_configs
from .model import B7Transformer
from .train import validate_settings
from .losses import flow_loss


def measure(cfg, *, batch_size, device="cuda", warmup=3, iterations=10, memory_budget_gib=35.):
    if any(type(v) is not int or v < 1 for v in (batch_size, warmup, iterations)):
        raise ValueError("Positive integer batch_size, warmup and iterations required")
    if not np.isfinite(memory_budget_gib) or memory_budget_gib <= 0:
        raise ValueError("Positive finite memory budget required")
    mc, lc, sc = validate_settings(cfg)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the server's BioFlow Python")
    torch.use_deterministic_algorithms(True)
    random.seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    torch.manual_seed(cfg["training"]["seed"])
    if cuda:
        torch.cuda.manual_seed_all(cfg["training"]["seed"])
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = {"status": "starting", "device": str(device), "torch": torch.__version__,
        "window_seconds": sc.window_seconds, "batch_size": batch_size, "warmup": warmup,
        "iterations": iterations, "dtype": "float32", "amp": False,
        "optimizer": "AdamW", "ema": True, "model_config": mc.to_dict(),
        "memory_budget_gib": memory_budget_gib, "scope": "synthetic training compute; no data loading or K16 evaluation",
        "gpu_free_before_gib": free / 2**30 if cuda else None,
        "gpu_total_gib": total / 2**30 if cuda else None}
    if cuda:
        result["gpu"] = torch.cuda.get_device_name(device)
    try:
        model = B7Transformer(mc).to(device).float().train()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        tr = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
        ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
        length = sc.window_seconds * sc.sample_rate
        batch = {"anchor": torch.randn(batch_size, 1, length, device=device),
                 "target": torch.randn(batch_size, 7, length, device=device),
                 "lead_mask": torch.ones(batch_size, 7, dtype=torch.bool, device=device),
                 "time_mask": torch.ones(batch_size, length, dtype=torch.bool, device=device),
                 "metadata": {"numeric": torch.zeros(batch_size, 3, device=device),
                              "sex": torch.full((batch_size,), 2, dtype=torch.long, device=device),
                              "field_mask": torch.zeros(batch_size, 4, dtype=torch.bool, device=device),
                              "age_topcoded": torch.zeros(batch_size, 1, device=device)}}
        original_anchor = batch["anchor"].clone()

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = flow_loss(model, batch, lc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    ema[key].mul_(tr["ema_decay"]).add_(value, alpha=1 - tr["ema_decay"])
            return loss.detach()

        for _ in range(warmup):
            update()
        if cuda:
            torch.cuda.synchronize(device)
            warm_allocated = torch.cuda.max_memory_allocated(device)
            warm_reserved = torch.cuda.max_memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(iterations):
            loss = update()
        if cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        # Include warmup/first AdamW allocation in capacity checks, not only steady state.
        allocated = max(warm_allocated, torch.cuda.max_memory_allocated(device)) / 2**30 if cuda else None
        reserved = max(warm_reserved, torch.cuda.max_memory_reserved(device)) / 2**30 if cuda else None
        finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        unchanged = torch.equal(original_anchor, batch["anchor"])
        if not finite or not unchanged:
            raise FloatingPointError("Nonfinite parameters or modified I condition")
        within = reserved <= memory_budget_gib if cuda else None
        result.update(status=("ok" if within else "over_budget") if cuda else "cpu_only",
            elapsed_seconds=elapsed, seconds_per_update=elapsed / iterations,
            windows_per_second=batch_size * iterations / elapsed,
            peak_allocated_gib=allocated, peak_reserved_gib=reserved,
            within_memory_budget=within, loss=float(loss), finite_parameters=True,
            finite_clipped_gradients=True, anchor_unchanged=True)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", within_memory_budget=False, error=str(exc))
        if cuda:
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, required=True)
    p.add_argument("--window-seconds", type=int, choices=(10, 40), required=True)
    p.add_argument("--batch-size", type=int, help="Defaults to the approved P64/long32 recipe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--memory-budget-gib", type=float, default=35.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new probe output; existing measurements are preserved")
    cfg = load_configs(args.preset)["public10" if args.window_seconds == 10 else "public40"]
    result = measure(cfg, batch_size=args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"],
        device=args.device, warmup=args.warmup, iterations=args.iterations, memory_budget_gib=args.memory_budget_gib)
    result["preset"] = args.preset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
