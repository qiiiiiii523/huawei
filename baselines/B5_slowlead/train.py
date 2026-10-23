"""Matched Huawei adaptation of P; G120/FM/history remain frozen in A/B/C."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import seed_all, device_from_name
from baselines.B5_best.common import write, sha, state_sha
from . import PROTOCOL, GROUPS
from .common import load_study, initial_model, load_model, Records
from .model import global_inputs
from .losses import objective
from .evaluate import Evaluator


def model_inputs(batch):
    return {"anchor": batch["anchor"], **global_inputs(batch)}


def optimizer_for(model, cfg):
    decoder = list(model.local.lead_decoder.parameters())
    ids = {id(p) for p in decoder}
    groups = [{"params": [p for p in model.local.parameters() if p.requires_grad and id(p) not in ids],
               "lr": cfg["cnn_learning_rate"]}]
    if model.group != "A":
        groups.append({"params": [p for p in decoder if p.requires_grad], "lr": cfg["decoder_learning_rate"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def run(a):
    root, output = Path(a.run).resolve(), Path(a.output).resolve()
    study = load_study(root)
    cfg, device = study["training"], device_from_name(a.device)
    seed_all(cfg["seed"], True)
    model = initial_model(root, study, device).set_group(a.group)
    scales = torch.tensor(study["scale_uV"], device=device)
    opt = optimizer_for(model, cfg)
    generator = torch.Generator().manual_seed(cfg["seed"])
    epoch, updates, best_r, best_row, best_location = 0, 0, None, None, None
    resumed = None
    if a.resume:
        if Path(a.resume).resolve() != output / "last.pt":
            raise ValueError("Resume this arm's own last.pt")
        model, resumed = load_model(a.resume, study, root / "study.json", device)
        if resumed["group"] != a.group or resumed["output_dir"] != str(output):
            raise ValueError("Wrong group/output for resume")
        selected_model, selected = load_model(output / "best_r.pt", study, root / "study.json", device)
        del selected_model
        if (selected["group"] != a.group or selected["output_dir"] != str(output)
                or selected["updates"] > resumed["updates"]):
            raise ValueError("Best checkpoint mismatch/newer than last; inspect interrupted saves")
        opt = optimizer_for(model, cfg)
        opt.load_state_dict(resumed["optimizer"])
        epoch, updates = resumed["epoch"], resumed["updates"]
        best_r, best_row, best_location = selected["best_r"], selected["validation"], selected["best_location"]
        generator.set_state(resumed["loader_rng"].cpu())
    else:
        output.mkdir(parents=True, exist_ok=False)
    saved_rng = pack_rng()
    try:
        evaluator = Evaluator(root, study, device)
    finally:
        restore_rng(saved_rng)
    try:
        loader = DataLoader(Records(study["source_run"], study["train"], study["scale_uV"][0]),
                            batch_size=cfg["batch_records"], shuffle=True, generator=generator, num_workers=0)

        def payload(validation=None):
            return dict(protocol=PROTOCOL, group=a.group, study_sha256=sha(root / "study.json"),
                source_g120_sha256=study["source_g120_sha256"], output_dir=str(output), stage="huawei", epoch=epoch,
                updates=updates, model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=opt.state_dict(), rng=pack_rng(), loader_rng=generator.get_state(),
                best_r=best_r, best_location=best_location, validation=validation,
                FM_optimizer_updates=0, context_optimizer_updates=0, global_optimizer_updates=0,
                local_optimizer_updates=updates)

        def evaluate_and_select():
            nonlocal best_r, best_row, best_location
            rng = pack_rng()
            folder = output / "validation" / f"epoch_{epoch:04d}"
            if folder.exists():
                folder = folder.with_name(folder.name + f"_retry_{time.time_ns()}")
            try:
                row = evaluator.evaluate(model, folder, a.group, epoch)
            finally:
                restore_rng(rng)
            if best_r is None or row["mean_r"] > best_r:
                best_r, best_row, best_location = row["mean_r"], row, {"stage": "huawei", "epoch": epoch}
                atomic_save(output / "best_r.pt", payload(row))
            return row

        if resumed is None:
            baseline = evaluate_and_select()
            if abs(baseline["mean_r"] - study["reference"]["mean_r"]) > 2e-6:
                raise ValueError("Epoch0 no longer reproduces source G120")
            write(output / "initialization.json", dict(group=a.group, initial_weights_sha256=state_sha(model.state_dict()),
                source_g120_sha256=study["source_g120_sha256"], source_epoch=study["source_g120_epoch"],
                trainable_CNN=True, trainable_G120=False, FM_fixed=True, context_fixed=True,
                II_stop_gradient=a.group == "C", training=cfg, epoch_zero_candidate=True, baseline=baseline))
            atomic_save(output / "last.pt", payload(baseline))
        else:
            restore_rng(resumed["rng"])
        for epoch in range(epoch + 1, cfg["epochs"] + 1):
            model.train()
            started, totals, count = time.perf_counter(), {}, 0
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                opt.zero_grad(set_to_none=True)
                loss, parts = objective(model(**model_inputs(b)), b, scales)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              cfg["gradient_clip"], error_if_nonfinite=True)
                opt.step()
                updates += 1
                n = len(b["anchor"])
                for k, v in {"loss": loss, **parts}.items():
                    totals[k] = totals.get(k, 0.) + float(v.detach()) * n
                count += n
            if not count:
                raise ValueError("Training split is empty")
            row = evaluate_and_select() if epoch % cfg["validate_every"] == 0 or epoch == cfg["epochs"] else None
            history = dict(stage="huawei", epoch=epoch, updates=updates,
                train={k: v / count for k, v in totals.items()}, validation=row,
                best_r=best_r, best_location=best_location, seconds=time.perf_counter() - started)
            atomic_save(output / "last.pt", payload(row))
            with (output / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(history, allow_nan=False) + "\n")
            print(history, flush=True)
        write(output / "completed.json", dict(completed=True, protocol=PROTOCOL, group=a.group,
            huawei_epochs=cfg["epochs"], updates=updates, best_r=best_r, best_location=best_location, best=best_row,
            selection="full120 fixed-original-context mean_r", FM_optimizer_updates=0,
            global_optimizer_updates=0, context_optimizer_updates=0))
    finally:
        evaluator.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute-training", action="store_true")
    a = p.parse_args()
    if not a.execute_training:
        p.error("Training requires --execute-training")
    torch.set_num_threads(2)
    run(a)


if __name__ == "__main__":
    main()
