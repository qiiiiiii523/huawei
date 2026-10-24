"""Run two independent, sample-budget-matched large U7 experiments on one GPU."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe(config_path, max_peak_gib=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from baselines.B7.train import validate_settings
    from baselines.B7.model import B7UNet
    from baselines.B7.losses import flow_loss
    from baselines.B7.flow import sample

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    mc, lc, sc = validate_settings(cfg)
    tr = cfg["training"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the training environment")
    torch.set_num_threads(4)
    torch.manual_seed(tr["seed"])
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    model = B7UNet(mc).cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    length, batch_size = sc.window_seconds * sc.sample_rate, tr["batch_size"]
    batch = {"anchor": torch.randn(batch_size, 1, length, device="cuda"),
             "target": torch.randn(batch_size, 7, length, device="cuda"),
             "lead_mask": torch.ones(batch_size, 7, dtype=torch.bool, device="cuda")}
    clean_i = batch["anchor"].clone()
    loss, parts = flow_loss(model, batch, lc)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema[key].mul_(tr["ema_decay"]).add_(value, alpha=1 - tr["ema_decay"])
    if not torch.equal(clean_i, batch["anchor"]):
        raise RuntimeError("Lead I was modified")
    evaluated = copy.deepcopy(model).eval()
    evaluated.load_state_dict(ema, strict=True)
    prediction = sample(evaluated, batch["anchor"][:sc.validation_batch_size],
                        [f"probe-{i}" for i in range(sc.validation_batch_size)], config=sc)
    if prediction.shape != (sc.validation_batch_size, 7, length) or not torch.isfinite(prediction).all():
        raise RuntimeError("Invalid generated waveform")
    torch.cuda.synchronize()
    if max_peak_gib is not None and torch.cuda.max_memory_reserved() / 2**30 > max_peak_gib:
        raise RuntimeError(f"Probe reserved peak exceeds {max_peak_gib} GiB; inspect GPU log before training")
    print(json.dumps({"status": "PASS", "parameters": parameters, "seconds": sc.window_seconds,
                      "batch": batch_size, "K": sc.samples, "loss": float(loss.detach()),
                      "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30}), flush=True)


def commands(args, batch):
    config_dir = ROOT / "configs" / f"u7_large_bs{batch}"
    result = []
    for stage in ("public10", "public40"):
        result.append((f"probe_{stage}", [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--probe-config", str(config_dir / f"b7_{stage}.json")]))
    result.append(("pipeline", [sys.executable, "-u", "-m", "baselines.B7.pipeline",
        "--data-root", str(args.data_root.resolve()), "--metadata", str(args.metadata.resolve()),
        "--ptb-root", str(args.ptb_root.resolve()), "--reuse-data-from", str(args.reuse_data_from.resolve()),
        "--output", str(args.output.resolve() / f"bs{batch}_seed42"),
        "--config-dir", str(config_dir), "--gpu-p", args.gpu, "--gpu-a", args.gpu,
        "--gpu-b", args.gpu, "--max-train", "1"]))
    return result


def execute(command, log_path, env):
    print(f"[START] {log_path.stem}; log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            return process.wait()
        except KeyboardInterrupt:
            # Signal the pipeline, which then stops its own training children.
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        import signal
                        process.send_signal(signal.SIGINT)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
            process.wait()
            raise


def run(args):
    groups = {batch: commands(args, batch) for batch in (32, 64)}
    if args.dry_run:
        for batch, jobs in groups.items():
            for name, command in jobs:
                print(json.dumps({"batch": batch, "job": name, "command": command}))
        return 0
    from baselines.B7.pipeline import acquire_lock
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output / ".large_experiments.lock")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    status = {"bs32": "pending", "bs64": "pending"}

    def save():
        temporary = args.output / "experiments_status.json.tmp"
        temporary.write_text(json.dumps({"experiments": status,
            "success": all(v == "done" for v in status.values())}, indent=2), encoding="utf-8")
        temporary.replace(args.output / "experiments_status.json")

    active = None
    try:
        for batch, jobs in groups.items():
            active = f"bs{batch}"
            status[active] = "running"
            save()
            for name, command in jobs:
                code = execute(command, args.output / f"bs{batch}_{name}.log", env)
                if code:
                    status[active] = "failed"
                    print(f"[FAILED] {active}/{name}: exit={code}; independent experiment continues", flush=True)
                    break
            else:
                status[active] = "done"
                print(f"[DONE] {active}", flush=True)
            save()
        return 0 if all(v == "done" for v in status.values()) else 1
    except KeyboardInterrupt:
        if active:
            status[active] = "interrupted"
        save()
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-config", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--data-root", type=Path, default=Path("/home/qht/huawei/data"))
    p.add_argument("--metadata", type=Path, default=Path("/home/qht/huawei/metadata"))
    p.add_argument("--ptb-root", type=Path, default=Path("/home/qht/huawei/data/PTB-Diagnostic"))
    p.add_argument("--reuse-data-from", type=Path, default=Path("/home/qht/huawei_B7/results/B7/auto_seed42"))
    p.add_argument("--output", type=Path, default=Path("results/B7-U7-large"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.probe_config:
        probe(args.probe_config)
    else:
        import signal
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        raise SystemExit(run(args))


if __name__ == "__main__":
    # The old CLI now starts the agreed M2/M3 suite, not the superseded batch comparison.
    from scripts.run_b7_structures import main as structures_main
    structures_main()
