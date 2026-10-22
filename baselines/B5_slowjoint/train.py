"""H: Huawei joint. P: public joint last -> fresh optimizer -> Huawei joint."""
import argparse
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import seed_all, device_from_name
from baselines.B5_best.common import read, write, sha, state_sha
from . import PROTOCOL, GROUPS
from .common import load_study, initial_model, load_model, Records
from .model import global_inputs, FUSION_MODES
from .losses import objective
from .evaluate import Evaluator


def model_inputs(batch):
    return {"anchor": batch["anchor"], **global_inputs(batch)}


def optimizer_for(model, cfg):
    groups = [
        {"params": model.local.parameters(), "lr": cfg["cnn_learning_rate"]},
        {"params": model.global_model.parameters(), "lr": cfg["global_learning_rate"]}]
    if getattr(model, "fusion", None) is not None:
        groups.append({"params": model.fusion.parameters(), "lr": cfg["global_learning_rate"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


@torch.no_grad()
def public_diagnostic(model, loader, scales, device):
    model.eval()
    total, count = 0., 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        _, parts = objective(model(**model_inputs(b)), b, scales, public=True)
        n = len(b["anchor"])
        total += float(parts["slow_pcc"]) * n
        count += n
    return {"mean_record_slow_pcc_loss": total / count, "records": count, "used_for_selection": False}


def run(a):
    root, output = Path(a.run).resolve(), Path(a.output).resolve()
    study = load_study(root)
    cfg, device = study["training"], device_from_name(a.device)
    if a.group == "P" and not study.get("public"):
        raise ValueError("P requires a completed real120 public cache")
    seed_all(cfg["seed"], True)
    mode = getattr(a, "fusion_mode", "add")
    model = initial_model(root, study, device, fusion_mode=mode)
    scales = torch.tensor(study["scale_uV"], device=device)
    stage, epoch, counts = "initial", 0, {"public": 0, "huawei": 0}
    best_r, best_row, best_location = None, None, None
    opt, rng_generators = None, {}
    for name in ("public", "huawei"):
        rng_generators[name] = torch.Generator().manual_seed(cfg["seed"])
    resumed = None
    if a.resume:
        if Path(a.resume).resolve() != output / "last.pt":
            raise ValueError("Resume this arm's own last.pt")
        model, resumed = load_model(a.resume, study, root / "study.json", device)
        if resumed["group"] != a.group or resumed["output_dir"] != str(output) or resumed.get("fusion_mode", "add") != mode:
            raise ValueError("Wrong group/output for resume")
        selected_model, selected = load_model(output / "best_r.pt", study, root / "study.json", device)
        del selected_model
        if selected["group"] != a.group or sum(selected["updates"].values()) > sum(resumed["updates"].values()):
            raise ValueError("Best checkpoint newer than last; inspect interrupted save before resuming")
        stage, epoch, counts = resumed["stage"], resumed["epoch"], resumed["updates"]
        best_r, best_row, best_location = selected["best_r"], selected["validation"], selected["best_location"]
        for k, g in rng_generators.items():
            g.set_state(resumed["loader_rng"][k].cpu())
    else:
        output.mkdir(parents=True, exist_ok=False)
    saved_rng = pack_rng()
    try:
        evaluator = Evaluator(root, study, device)
    finally:
        restore_rng(saved_rng)
    try:
        loaders = {"huawei": DataLoader(Records(study["source_run"], study["train"], study["scale_uV"][0]),
                   batch_size=cfg["batch_records"], shuffle=True, generator=rng_generators["huawei"], num_workers=0)}
        if a.group == "P":
            public_root = study["public"]["root"]
            manifest = read(Path(public_root) / "public.json")
            loaders["public"] = DataLoader(Records(public_root, manifest["train"], study["scale_uV"][0]),
                batch_size=cfg["batch_records"], shuffle=True, generator=rng_generators["public"], num_workers=0)
            public_val = DataLoader(Records(public_root, manifest["validation"], study["scale_uV"][0]),
                batch_size=cfg["batch_records"], shuffle=False, num_workers=0)

        def payload(validation=None):
            return dict(protocol=PROTOCOL, group=a.group, fusion_mode=mode, study_sha256=sha(root / "study.json"),
                source_g120_sha256=study["source_g120_sha256"], output_dir=str(output), stage=stage, epoch=epoch,
                updates=dict(counts), model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=opt.state_dict() if opt else None, rng=pack_rng(),
                loader_rng={k: g.get_state() for k, g in rng_generators.items()},
                best_r=best_r, best_location=best_location, validation=validation,
                FM_optimizer_updates=0, context_optimizer_updates=0, local_optimizer_updates=sum(counts.values()),
                global_optimizer_updates=sum(counts.values()))

        def evaluate_and_select():
            nonlocal best_r, best_row, best_location
            rng = pack_rng()
            folder = output / "validation" / f"{stage}_{epoch:04d}"
            if folder.exists():
                folder = folder.with_name(folder.name + f"_retry_{time.time_ns()}")
            try:
                row = evaluator.evaluate(model, folder, a.group, epoch)
            finally:
                restore_rng(rng)
            if best_r is None or row["mean_r"] > best_r:
                best_r, best_row, best_location = row["mean_r"], row, {"stage": stage, "epoch": epoch}
                atomic_save(output / "best_r.pt", payload(row))
            return row

        if resumed is None:
            baseline = evaluate_and_select()
            if abs(baseline["mean_r"] - study["reference"]["mean_r"]) > 2e-6:
                raise ValueError("Epoch0 no longer reproduces source G120")
            write(output / "initialization.json", dict(group=a.group, initial_weights_sha256=state_sha(model.state_dict()),
                source_g120_sha256=study["source_g120_sha256"], source_epoch=study["source_g120_epoch"],
                fusion_mode=mode, fusion_after_block=6 if mode == "mid" else None,
                fusion_parameters=sum(p.numel() for p in model.fusion.parameters()) if getattr(model, "fusion", None) is not None else 0,
                trainable_CNN=True, trainable_G120=True, FM_fixed=True, context_fixed=True,
                training=cfg, epoch_zero_candidate=True, baseline=baseline))
            atomic_save(output / "last.pt", payload(baseline))
        phases = (["public"] if a.group == "P" else []) + ["huawei"]
        if resumed:
            if stage not in ("initial", *phases):
                raise ValueError("Unknown resume phase")
            restore_rng(resumed["rng"])
        for phase in phases:
            if resumed and stage != "initial" and phases.index(phase) < phases.index(stage):
                continue
            same_stage = stage == phase
            first = epoch + 1 if same_stage else 1
            opt = optimizer_for(model, cfg)
            if same_stage and resumed:
                opt.load_state_dict(resumed["optimizer"])
            if not same_stage:
                stage, epoch = phase, 0
                # Both H and P reset to the identical Huawei shuffle order.
                rng_generators[phase].manual_seed(cfg["seed"])
                if phase == "huawei":
                    seed_all(cfg["seed"], True)
                    transition = evaluate_and_select()  # also test public-last before Huawei updates
                else:
                    transition = None
                atomic_save(output / "last.pt", payload(transition))
            budget = cfg["public_epochs"] if phase == "public" else cfg["epochs"]
            for epoch in range(first, budget + 1):
                model.train()
                started, totals, count = time.perf_counter(), {}, 0
                for b in loaders[phase]:
                    b = {k: v.to(device) for k, v in b.items()}
                    opt.zero_grad(set_to_none=True)
                    loss, parts = objective(model(**model_inputs(b)), b, scales, public=phase == "public")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                    opt.step()
                    counts[phase] += 1
                    n = len(b["anchor"])
                    for k, v in {"loss": loss, **parts}.items():
                        totals[k] = totals.get(k, 0.) + float(v.detach()) * n
                    count += n
                row, diagnostic = None, None
                if epoch % cfg["validate_every"] == 0 or epoch == budget:
                    if phase == "huawei":
                        row = evaluate_and_select()
                    else:
                        rng = pack_rng()
                        try:
                            diagnostic = public_diagnostic(model, public_val, scales, device)
                        finally:
                            restore_rng(rng)
                history = dict(stage=phase, epoch=epoch, fusion_mode=mode, updates=dict(counts),
                    train={k: v / count for k, v in totals.items()}, validation=row, public_validation=diagnostic,
                    best_r=best_r, best_location=best_location, seconds=time.perf_counter() - started)
                atomic_save(output / "last.pt", payload(row))
                # Per-epoch entries survive a terminal disconnect; last.pt is saved first.
                import json
                with (output / "history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(history, allow_nan=False) + "\n")
                print(history, flush=True)
            resumed = None
        write(output / "completed.json", dict(completed=True, protocol=PROTOCOL, group=a.group, fusion_mode=mode,
            huawei_epochs=cfg["epochs"], public_epochs=cfg["public_epochs"] if a.group == "P" else 0,
            updates=counts, best_r=best_r, best_location=best_location, best=best_row,
            selection="full120 fixed-context mean_r", FM_optimizer_updates=0, context_optimizer_updates=0))
    finally:
        evaluator.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--fusion-mode", choices=FUSION_MODES, default="add")
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute-training", action="store_true")
    a = p.parse_args()
    if not a.execute_training:
        p.error("Training requires --execute-training on the server")
    torch.set_num_threads(2)
    run(a)


if __name__ == "__main__":
    main()
