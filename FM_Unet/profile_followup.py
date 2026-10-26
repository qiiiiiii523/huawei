"""Bounded real-data memory profile: batch64 AdamW/EMA plus Heun16/K16 and frozen fast."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from .experiment import load_experiment, build_optimizer
from .config import LossConfig
from .train import load_statistics
from .train_data import read_manifest, load_record, native_target
from .train_eval import sample_window, FrozenFast
from .train_loss import flow_loss

MANIFESTS = {
    'public10': '/home/qht/huawei/data_cache/B7_public_qcv2/data/public10/manifest.json',
    'public75': '/home/qht/huawei_B7/results/B7/auto_seed42/ptb_recovery/data/manifest.json',
    'huawei120': '/home/qht/huawei_B7/results/B7/auto_seed42/data/huawei/manifest.json',
}
FAST_CODE = '/home/qht/huawei_B5_final'
FAST_CHECKPOINT = '/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt'


def process_memory_mib():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    memory = []
    for line in result.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) == 2 and fields[0] == str(os.getpid()) and fields[1].isdigit():
            memory.append(int(fields[1]))
    return sum(memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('wide3', 'adaptive7'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    scales, noise = load_statistics(Path(__file__).with_name('best.pt'))
    reports = []
    root = Path(__file__).resolve().parents[1]
    for stage, manifest_path in MANIFESTS.items():
        cfg, model = load_experiment(root/'configs/fm_unet_followup'/args.variant/f'{stage}.json')
        manifest = read_manifest(manifest_path)
        row = next(r for r in manifest['records'] if r['split'] == 'train' and r['input_reliable']
                   and r.get('real_samples', 0) >= model.input_seconds*500)
        if stage == 'huawei120':
            import sys
            sys.path.insert(0, FAST_CODE)
            from baselines.B5.metadata import DemographicsTable
            row['metadata'] = DemographicsTable('/home/qht/huawei/data/userinfobean.csv').get(row['subject_id'])
        raw, target, mask, metadata = load_record(row)
        points = model.input_seconds*500
        raw = raw[:, :points]
        native = native_target(target, mask, scales, model.rate)[:, :model.input_seconds*model.rate]
        model = model.cuda().train()
        optimizer = build_optimizer(model, cfg)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        batch = dict(anchor=torch.as_tensor(raw/scales[0], device='cuda', dtype=torch.float32)[None].repeat(64, 1, 1),
                     target=torch.as_tensor(native, device='cuda')[None].repeat(64, 1, 1),
                     lead_mask=torch.as_tensor(mask[[1,6,7,8,9,10,11]], device='cuda')[None].repeat(64, 1),
                     metadata={k: torch.as_tensor(v, device='cuda')[None].expand(64, *np.shape(v)) for k, v in metadata.items()})
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = flow_loss(model, batch, LossConfig(**cfg['loss']), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for k, value in model.state_dict().items():
                    ema[k].mul_(.999).add_(value, alpha=.001)
        torch.cuda.synchronize()
        train_process = process_memory_mib()
        evaluated = copy.deepcopy(model).eval()
        evaluated.load_state_dict(ema, strict=True)
        result = sample_window(evaluated, raw, metadata, scales, noise, 'bounded-profile', cfg['validation'])
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite profiled slow output')
        fast = FrozenFast(FAST_CODE, FAST_CHECKPOINT, 'cuda') if stage == 'huawei120' else None
        if fast is not None:
            prediction = fast.predict(raw, metadata, 'bounded-profile')
            if not np.isfinite(prediction+result).all():
                raise ValueError('Nonfinite profiled complete output')
        torch.cuda.synchronize()
        report = dict(variant=args.variant, stage=stage, batch=64, optimizer_updates=2,
                      allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                      reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                      sampled_process_mib=max(train_process, process_memory_mib()),
                      parameters=sum(p.numel() for p in model.parameters()),
                      samples=16, solver='heun', steps=16, frozen_fast=fast is not None)
        reports.append(report)
        print(json.dumps(report), flush=True)
        del batch, loss, parts, evaluated, fast, model, optimizer, ema
        torch.cuda.empty_cache()
    (output/'profile.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    (output/'completed.json').write_text(json.dumps(dict(completed=True, variant=args.variant)), encoding='utf-8')


if __name__ == '__main__':
    main()
