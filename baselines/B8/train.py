"""Direct M1-regression training on B7's manifests, frozen scales and full120 selection."""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from baselines.B7.data import WindowDataset, read_manifest
from baselines.B7.train import balanced_indices
from . import ARCHITECTURE_ID, CHECKPOINT_VERSION, GENERATED_LEADS
from .checkpoint import load, save, model_from_checkpoint
from .evaluate import evaluate
from .losses import regression_loss
from .model import B8M1TimeRegressor

def move(value, device): return {k: move(v, device) if isinstance(v, dict) else v.to(device) for k, v in value.items()}

def parse_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if set(cfg) != {"stage", "model", "loss", "training"} or cfg["stage"] not in {"public10", "huawei10"}: raise ValueError("B8 config requires stage/model/loss/training")
    model = cfg["model"]
    if set(model) - {"d_model", "blocks", "heads", "ffn", "dropout"}: raise ValueError("unknown B8 model setting")
    tr = cfg["training"]
    for name in ("seed", "updates", "validate_every", "batch_size", "learning_rate", "weight_decay", "ema_decay", "gradient_clip", "boundary_probability"):
        if name not in tr: raise ValueError("missing training setting: " + name)
    if tr["updates"] < 1 or tr["validate_every"] < 1 or tr["batch_size"] < 1 or tr["updates"] % tr["validate_every"]: raise ValueError("invalid B8 training budget")
    if not 0 <= tr["ema_decay"] < 1 or tr["learning_rate"] <= 0 or tr["gradient_clip"] <= 0: raise ValueError("invalid optimizer settings")
    loss = cfg["loss"]
    if loss.get("pcc_leads") != 11 or loss.get("ii_weight") != 5: raise ValueError("B8 retains B7 PCC11 and II=5 policy")
    return cfg

def batch_loss(model, batch, loss):
    target = batch["target"]
    prediction = model(batch["anchor"])
    return regression_loss(prediction, target, batch["lead_mask"], batch["time_mask"], mae_weight=loss["mae_weight"], pcc_weight=loss["pcc_weight"], ii_weight=loss["ii_weight"], variance_floor=loss["target_variance_floor"])

def payload(model, optimizer, ema, cfg, manifest, step, best):
    return {"architecture_id": ARCHITECTURE_ID, "format_version": CHECKPOINT_VERSION, "architecture": model.architecture,
            "generated_leads": GENERATED_LEADS, "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
            "config": cfg, "scales_uV": manifest["scales_uV"], "step": step, "best_r": best,
            "selection": "B7-complete120-missing11-macro" if cfg["stage"] == "huawei10" else "public-record-missing11"}

def train(args):
    cfg, manifest = parse_config(args.config), read_manifest(args.manifest)
    if cfg["stage"] == "huawei10" and manifest.get("huawei_window_protocol") is None: raise ValueError("B8 Huawei training needs the B7 fixed Huawei10 view")
    if args.output.exists(): raise FileExistsError("new B8 output directory required")
    random.seed(cfg["training"]["seed"]); np.random.seed(cfg["training"]["seed"]); torch.manual_seed(cfg["training"]["seed"])
    torch.use_deterministic_algorithms(True); device = torch.device(args.device)
    model = B8M1TimeRegressor(**cfg["model"]).to(device)
    if args.init_checkpoint:
        source = load(args.init_checkpoint)
        if source["architecture"] != model.architecture or source["scales_uV"] != manifest["scales_uV"]: raise ValueError("B8 initialization must preserve architecture and frozen scales")
        model.load_state_dict(source["ema"], strict=True)
    dataset = WindowDataset(manifest, 10, cfg["training"]["boundary_probability"], include_d12=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}; args.output.mkdir(parents=True)
    best = -float("inf")
    for step in range(1, cfg["training"]["updates"] + 1):
        model.train(); batch = move(default_collate([dataset[i] for i in balanced_indices(dataset, cfg["training"]["batch_size"])]), device)
        optimizer.zero_grad(set_to_none=True); total, parts = batch_loss(model, batch, cfg["loss"]); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True); optimizer.step()
        with torch.no_grad():
            for key, value in model.state_dict().items(): ema[key].mul_(cfg["training"]["ema_decay"]).add_(value, alpha=1 - cfg["training"]["ema_decay"])
        if step % cfg["training"]["validate_every"] == 0 or step == cfg["training"]["updates"]:
            evaluated = copy.deepcopy(model).eval(); evaluated.load_state_dict(ema, strict=True)
            metrics, rows = evaluate(evaluated, manifest, cfg["stage"])
            current = payload(model, optimizer, ema, cfg, manifest, step, best)
            if metrics["r"] > best: best = metrics["r"]; current["best_r"] = best; save(args.output / "best_r.pt", current)
            save(args.output / "last.pt", payload(model, optimizer, ema, cfg, manifest, step, best))
            (args.output / f"validation_{step:07d}.json").write_text(json.dumps({"metrics": metrics, "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"step": step, "loss": float(total.detach()), "mae": float(parts["mae"]), "r": metrics["r"]}, ensure_ascii=False), flush=True)
    (args.output / "completed.json").write_text(json.dumps({"completed": True, "best_r": best, "updates": cfg["training"]["updates"]}), encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest", required=True); p.add_argument("--config", required=True); p.add_argument("--output", required=True, type=Path); p.add_argument("--device", default="cpu"); p.add_argument("--init-checkpoint")
    train(p.parse_args())
if __name__ == "__main__": main()
