"""Prepare two portable, separate P handoffs; tar/upload are explicit next steps."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_b7_a1 import read_parent, make_config

EXPECTED = {"B7_M2_P": ("baselines.B7", 11477663),
            "B7T_g6_19m_P": ("baselines.B7T_A1", 19398702)}


def inspect_parent(path, name):
    module, parent = read_parent(path)
    expected_module, count = EXPECTED[name]
    checkpoint = importlib.import_module(module + ".checkpoint")
    model = checkpoint.model_from_checkpoint(parent)
    actual = sum(p.numel() for p in model.parameters())
    if module != expected_module or actual != count:
        raise ValueError(f"Wrong {name} architecture: {module}, parameters={actual}; expected {count}")
    budget = parent["identity"]["config"]["training"]["updates"]
    if parent["update"] != budget:
        raise ValueError(f"{name}: selected checkpoint is step {parent['update']}, not final step {budget}")
    return module, parent, actual


def find_t_parent():
    matches = []
    for root in Path("/home/qht").glob("huawei_B7_T*"):
        if not root.is_dir():
            continue
        for path in root.rglob("best_r.pt"):
            if path.parent.name != "P" or not (path.parent / "completed.json").is_file():
                continue
            config_path = path.parent / "config.json"
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("model", {})
            if model.get("depths") == [2, 2, 6] and model.get("embed_dims") == [128, 256, 384]:
                matches.append(path.resolve())
    if not matches:
        raise FileNotFoundError("No completed g6_19m P found; specify --t-checkpoint /exact/path/P/best_r.pt")
    from baselines.B7.checkpoint import file_sha256
    unique = {}
    for path in matches:
        unique.setdefault(file_sha256(path), []).append(path)
    if len(unique) != 1:
        raise ValueError("Multiple different g6 P weights found; choose --t-checkpoint explicitly:\n" +
                         "\n".join(map(str, matches)))
    return sorted(matches)[0]


def environment():
    import torch
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch_build_cuda": torch.version.cuda}
    for package in ("torch", "numpy", "scipy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def prepare_bundle(path, name, target, inspected):
    module, parent, count = inspected
    from baselines.B7.checkpoint import file_sha256
    target.mkdir()
    (target / "P").mkdir()
    shutil.copy2(path, target / "P/best_r.pt")
    shutil.copy2(path.parent / "completed.json", target / "P/completed.json")
    cfg = parent["identity"]["config"]
    (target / "P/config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # Frozen scales are already in the checkpoint; this copy makes the loader contract explicit.
    contract = {"sample_rate": 500, "raw_unit": "uV", "scales_uV": parent["scales_uV"],
        "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "generated_leads": list(parent["generated_leads"]), "I_noise": False,
        "normalization": "fixed per-lead division; do not re-fit scales on Huawei validation/test"}
    (target / "P/data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    # Keep only code, not .git, caches, model duplicates, logs or private data.
    (target / "baselines").mkdir()
    (target / "baselines/__init__.py").write_text("", encoding="utf-8")
    packages = ["B7"] + (["B7T_A1"] if module.endswith("B7T_A1") else [])
    for package in packages:
        dest = target / "baselines" / package
        dest.mkdir()
        for source in sorted((ROOT / "baselines" / package).glob("*.py")):
            shutil.copy2(source, dest / source.name)
    (target / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/run_b7_a1.py", target / "scripts/run_b7_a1.py")
    (target / "requirements.txt").write_text("numpy>=1.24\ntorch>=2.0\nscipy>=1.10\n", encoding="utf-8")
    (target / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    args = argparse.Namespace(seed=42, updates=1000, checkpoint_every=500, batch_size=32,
                              learning_rate=1e-5, ode_batch=16, validation_records=4)
    short = make_config(module, parent, args)
    configs = target / "configs"
    configs.mkdir()
    (configs / "finetune_huawei10.json").write_text(json.dumps(short, indent=2), encoding="utf-8")
    long = json.loads(json.dumps(short))
    long["stage"] = "huawei40"
    long["sampling"]["window_seconds"] = 40
    long["training"]["boundary_probability"] = .15
    (configs / "finetune_huawei40.json").write_text(json.dumps(long, indent=2), encoding="utf-8")
    public = json.loads(json.dumps(long))
    public["stage"] = "public40"
    (configs / "continue_ptb40.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    source_score = path.parent / f"validation_{parent['update']:07d}.json"
    metrics = json.loads(source_score.read_text(encoding="utf-8")).get("metrics") if source_score.is_file() else None
    info = {"name": name, "backend": module, "parameters": count, "stage": "public10",
        "update": parent["update"], "public_validation_r": parent["best_r"], "metrics": metrics,
        "source_checkpoint": str(path.resolve()), "checkpoint_sha256": file_sha256(path),
        "source_code_sha256": parent["identity"].get("code_sha256"),
        "source_manifest_sha256": parent["identity"].get("manifest_sha256"),
        "parent_model_config": parent["model_config"], "original_P_identity_config": cfg,
        "handoff_code": "B7-A1 runtime; same architecture, supports new Huawei10 or Huawei40 stage",
        "new_stage_optimizer": "fresh AdamW; initialize model from P EMA; do not --resume P",
        "data_included": False}
    (target / "handoff.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    readme = f"""# {name} P权重交接

模型参数量 {count:,}；P最终更新 {parent['update']}；公开验证r {parent['best_r']:.9f}。
这是公开P权重，不是华为A/A1/B权重。后续模型和任务变更由接收方自行训练。

必需内容：P/best_r.pt（原完整checkpoint，内含EMA/结构/loss/尺度/初始化历史）、
对应baselines模型和训练代码、requirements.txt。P/config.json、data_contract.json、
completed.json及handoff.json用于核对配置/单位/完成状态与来源。
只发一份权重；last.pt/best.pt/initial.pt均不重复打包。checkpoint原样保留，
文件内即使含P optimizer/RNG，新阶段也不加载它们；仅用P EMA初始化。

## 接收方准备

解压后cd到本目录，准备含PyTorch的Python环境，依赖参考requirements.txt和environment.json。
环境版本清单不保证其他CUDA/驱动组合可用，CPU可以核验加载；实际训练需适配的GPU环境。
`sha256sum -c SHA256SUMS`核验文件。
运行命令必须从本目录根执行，不要与另一包的baselines混放，也不要在活动的原实验目录覆盖代码。

原始数据和缓存没有打包。接收方必须拥有合法可用的训练/验证数据。
输入是同步完整I，目标为同记录12导联，500Hz、原始微伏uV；不能把独立手表记录直接
当作逐点对齐I。清洗后使用P/data_contract.json的固定尺度，不在验证/测试上拟合归一化。
若复现同一华为结果，还需同样患者划分、QC、配对分组与metadata编码。

本代码读取B7 manifest：protocol=B7-raw-uV-records-v1，sample_rate=500，
scales_uV与P相同，records各项含record_id/subject_id/target_record_id/source/group/path/split/input_reliable。
NPZ含anchor_uV[1,T]、target_uV[12,T]、lead_mask[12]；可有完整人口学数组。
path为相对manifest的本地路径或当地有效绝对路径，不能沿用发送方服务器绝对路径。
华为验证使用完整真实120秒、三组task1/machine/body与固定QC，评分权重0.5/0.25/0.25。
模型支持7通道生成并按公式恢复12导联，I不加噪。

## 直接开始新阶段（示例GPU0，自备manifest）

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m {module}.train \\
  --manifest /接收方/华为/manifest.json \\
  --config configs/finetune_huawei40.json \\
  --init-checkpoint P/best_r.pt --output runs/A40 --device cuda
```

40秒适配的FM/MAE全窗、PCC中央15–25秒；推理每次保留中央10秒拼回完整记录。
10秒适配将config改为configs/finetune_huawei10.json、output改为runs/A10。
该模式生成完整10秒，所有loss全10秒，拼回120秒再评分。
PTBDB继续训练可使用configs/continue_ptb40.json及对应本地PTBDB manifest。
随后华为B应从Q的新权重初始化，不再使用P作为父权重。

这些示例默认batch32、1000步、lr1e-5、每500步保存、仅末尾K16/Heun16验证。
接收方可在首次新训练前设定新配置；不要修改已经启动的实验配置后直接resume。
`--init-checkpoint`是新阶段，`--resume`只用于接收方自己已有last.pt的同一阶段。
训练中断后用相同config和output改为`--resume`，去掉`--init-checkpoint`。

也可用scripts/run_b7_a1.py自动识别模型，只做10秒华为适配：
```bash
python -u scripts/run_b7_a1.py --parent-checkpoint P/best_r.pt \\
  --manifest /接收方/华为/manifest.json --output runs/A1 --gpu 0
```
"""
    (target / "README.md").write_text(readme, encoding="utf-8")
    entries = []
    for file in sorted(target.rglob("*")):
        if file.is_file():
            entries.append(file_sha256(file) + "  " + file.relative_to(target).as_posix())
    (target / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return info


def run(args):
    m2 = args.m2_checkpoint.resolve()
    t = args.t_checkpoint.resolve() if args.t_checkpoint else find_t_parent()
    # Inspect both before creating staging output; no partial wrong-model handoff.
    inspected = {"B7_M2_P": inspect_parent(m2, "B7_M2_P"),
                 "B7T_g6_19m_P": inspect_parent(t, "B7T_g6_19m_P")}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, path in (("B7_M2_P", m2), ("B7T_g6_19m_P", t)):
        info = prepare_bundle(path, name, output / name, inspected[name])
        print(json.dumps({k: info[k] for k in ("name", "parameters", "update", "public_validation_r", "source_checkpoint")}), flush=True)
    print("Staging directory: " + str(output), flush=True)
    print("No tar/upload performed; source checkpoints and datasets were not changed.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m2-checkpoint", type=Path, default=Path("/home/qht/huawei_B7_U7_large/results/B7-M2-M3_gpu2/M2_seed42/P/best_r.pt"))
    p.add_argument("--t-checkpoint", type=Path, help="Optional exact completed g6_19m P; otherwise discover under /home/qht/huawei_B7_T*")
    p.add_argument("--output", type=Path, default=ROOT / "results/P_handoffs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
