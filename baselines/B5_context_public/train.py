"""Matched record-uniform public -> Huawei FM training with EMA and exact epoch resume."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.common import read, write, sha
from . import PROTOCOL
from .sampling import SAMPLING, validate_machine_references
from .data import load_study, read_record, tensor_batch
from .common import create_model
from .evaluate import evaluate
from .flow import flow_loss


def run(a):
    root, output = Path(a.run).resolve(), Path(a.output).resolve()
    study = load_study(root); cfg = study["training"]
    refs_path = Path(a.references).resolve(); refs = read(refs_path)
    if not refs.get("completed") or refs["study_sha256"] != sha(root / "study.json"):
        raise ValueError("Run matching paired references before training")
    validate_machine_references(refs["reports"])
    device = device_from_name(a.device); seed_all(cfg["seed"], True)
    model = create_model(study, a.arm, device)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scale = np.asarray(study["scale_uV"], np.float32)
    stages = ("public", "huawei")
    stage, epoch, updates = "initial", 0, {s: 0 for s in stages}
    best_r = refs["reports"]["context_public_initial" if a.arm == "context_public" else "control_initial"]["machine_r"]
    best_location, selected_row = dict(stage="initial", epoch=0), None
    rng = np.random.default_rng(cfg["seed"])
    optimizer = None
    resumed = None
    if a.resume:
        if Path(a.resume).resolve() != output / "last.pt": raise ValueError("Resume this output's last.pt")
        resumed = torch.load(a.resume, map_location="cpu", weights_only=True)
        if (resumed.get("protocol") != PROTOCOL or resumed["study_sha256"] != sha(root / "study.json")
            or resumed["sampling"] != SAMPLING or resumed["arm"] != a.arm or resumed["output_dir"] != str(output)
            or resumed["references_sha256"] != sha(refs_path)):
            raise ValueError("Resume study/arm/output/sampling/references mismatch")
        model.load_state_dict(resumed["model"])
        ema = {k: v.to(device) for k, v in resumed["ema"].items()}
        stage, epoch, updates = resumed["stage"], resumed["epoch"], resumed["updates"]
        selected = torch.load(output / "best_r.pt", map_location="cpu", weights_only=True)
        if selected["study_sha256"] != resumed["study_sha256"] or selected["arm"] != a.arm:
            raise ValueError("Best belongs to a different experiment")
        if sum(selected["updates"].values()) > sum(updates.values()):
            raise ValueError("Best is newer than last; interrupted best/last save needs inspection")
        best_r, best_location, selected_row = selected["best_r"], selected["best_location"], selected["validation"]
        rng.bit_generator.state = json.loads(resumed["numpy_generator_state"])
        restore_rng(resumed["rng"])
    else:
        output.mkdir(parents=True, exist_ok=False)

    def payload(validation=None):
        return dict(protocol=PROTOCOL, arm=a.arm, output_dir=str(output), sampling=SAMPLING,
            study_sha256=sha(root / "study.json"), references_sha256=sha(refs_path), stage=stage, epoch=epoch,
            updates=dict(updates), model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            ema={k: v.detach().cpu().clone() for k, v in ema.items()},
            optimizer=optimizer.state_dict() if optimizer else None, rng=pack_rng(),
            numpy_generator_state=json.dumps(rng.bit_generator.state), best_r=best_r, best_location=best_location,
            validation=validation, selection_metric="machine_full120_hybrid_r")

    def assess():
        nonlocal best_r, best_location, selected_row
        saved_rng = pack_rng()
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(ema); model.eval()
            row = evaluate(model, root, study, output / "validation" / f"{stage}_{epoch:04d}_{time.time_ns()}", a.arm == "context_public")
        finally:
            model.load_state_dict(original); restore_rng(saved_rng)
        if row["machine_r"] > best_r:
            best_r, best_location, selected_row = row["machine_r"], dict(stage=stage, epoch=epoch), row
            atomic_save(output / "best_r.pt", payload(row))
        return row

    if resumed is None:
        # Seed and initial source have been checked by references. Keep epoch0 as a candidate.
        selected_row = refs["reports"]["context_public_initial" if a.arm == "context_public" else "control_initial"]
        atomic_save(output / "best_r.pt", payload(selected_row))
        atomic_save(output / "last.pt", payload(selected_row))
    for phase in stages:
        if stage != "initial" and stages.index(phase) < stages.index(stage): continue
        same = stage == phase
        first = epoch + 1 if same else 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        if same and resumed:
            optimizer.load_state_dict(resumed["optimizer"])
        if not same:
            if phase == "huawei": model.load_state_dict(ema)  # public fixed-budget last EMA, no public best selection
            stage, epoch = phase, 0
            rng = np.random.default_rng(cfg["seed"])
            seed_all(cfg["seed"], True)
            transition = assess() if phase == "huawei" else None
            atomic_save(output / "last.pt", payload(transition))
        entries = study["records"][phase + "_train"]
        budget = cfg[phase + "_epochs"]
        for epoch in range(first, budget + 1):
            started = time.perf_counter(); model.train()
            order = rng.permutation(len(entries)); total, count = 0., 0
            for offset in range(0, len(order), cfg["batch_records"]):
                selected = order[offset:offset + cfg["batch_records"]]
                records = [read_record(root, entries[int(i)]) for i in selected]
                starts = [int(rng.integers(r["target_uV"].shape[-1] // 5000)) * 5000 for r in records]
                batch = tensor_batch(records, starts, scale, device, a.arm == "context_public")
                optimizer.zero_grad(set_to_none=True)
                loss, parts = flow_loss(model, batch, study["loss"], torch.as_tensor(scale, device=device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step(); updates[phase] += 1
                with torch.no_grad():
                    for k, v in model.state_dict().items(): ema[k].mul_(cfg["ema_decay"]).add_(v, alpha=1 - cfg["ema_decay"])
                total += float(loss.detach()) * len(records); count += len(records)
            validation = assess() if phase == "huawei" and (epoch % cfg["validate_every"] == 0 or epoch == budget) else None
            record = dict(stage=phase, epoch=epoch, updates=dict(updates), loss=total / count,
                validation=validation, best_r=best_r, best_location=best_location, seconds=time.perf_counter() - started)
            atomic_save(output / "last.pt", payload(validation))
            with (output / "history.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(record) + "\n")
            print(record, flush=True)
        resumed = None
    write(output / "completed.json", dict(completed=True, protocol=PROTOCOL, arm=a.arm, updates=updates,
        public_epochs=cfg["public_epochs"], huawei_epochs=cfg["huawei_epochs"], best_r=best_r,
        best_location=best_location, sampling=SAMPLING, study_sha256=sha(root / "study.json")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "references", "output"): p.add_argument("--" + k, required=True)
    p.add_argument("--arm", choices=("control", "context_public"), required=True)
    p.add_argument("--resume"); p.add_argument("--device", default="cuda")
    p.add_argument("--execute-training", action="store_true")
    a = p.parse_args()
    if not a.execute_training: p.error("Add --execute-training to start optimizer updates")
    torch.set_num_threads(2); run(a)


if __name__ == "__main__": main()
