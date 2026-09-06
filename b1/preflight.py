"""Read-only data access and synthetic CPU timing; NOT an experiment result."""
import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from ecg12gen.dataset import UnifiedECGDataset
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.body_scale import BodyScaleVariantDataset
from ecg12gen.training import seed_everything
from ecg12gen.losses import masked_huber_loss, masked_pcc_loss
from b1.data import ROOT, local_config
from b1.model import LightweightUNet


def main():
    out = ROOT / 'results/b1/preflight'
    out.mkdir(parents=True, exist_ok=True)
    config = local_config()
    result = {'kind': 'synthetic_timing_not_validation_score', 'datasets': {},
              'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
              'paths': config.raw['paths'], 'timings': [],
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    for task in ('task1', 'task2'):
        for split in ('train', 'validation'):
            ds = UnifiedECGDataset(config, task, split)
            for sample in ds:
                sample.validate()
                assert np.isfinite(sample.X_ecg).all() and np.isfinite(sample.Y_12lead).all()
            result['datasets'][f'{task}_{split}'] = {'windows': len(ds), 'excluded': ds.excluded_rows}
    strict = StrictD12PretrainDataset(config, 'd12_six_pretrain')
    assert len({r['dedup_key'] for r in strict.rows}) == len(strict)
    for sample in strict:
        sample.validate()
    result['datasets']['strict_train'] = len(strict)
    for split in ('train', 'validation'):
        a = BodyScaleVariantDataset(config, split, 'A_raw_window')
        b = BodyScaleVariantDataset(config, split, 'B_detrend_0p2Hz_then_window')
        am = {s.meta['canonical_array_index']: s for s in a}
        assert set(am) == {s.meta['canonical_array_index'] for s in b}
        for s in b:
            ref = am[s.meta['canonical_array_index']]
            assert ref.meta['subject_id'] == s.meta['subject_id']
            assert np.array_equal(ref.Y_12lead, s.Y_12lead)
        result['datasets'][f'body_{split}'] = len(a)
    result['split_sha256'] = hashlib.sha256(config.path('subject_split_csv').read_bytes()).hexdigest()
    print('DATA PASS', result['datasets'], flush=True)
    for threads in (4, 8):
        torch.set_num_threads(threads)
        for channels in (1, 6):
            seed_everything(42, True)
            model = LightweightUNet(channels)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            x, target = torch.randn(16, channels, 5000), torch.randn(16, 12, 5000)
            mask = torch.ones(16, 12, dtype=torch.bool)
            mask[:, :channels] = False
            elapsed = []
            for step in range(5):
                begin = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                assert pred.shape == target.shape
                loss = masked_huber_loss(pred, target, mask) + .1 * masked_pcc_loss(pred, target, mask)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step()
                if step:
                    elapsed.append(time.perf_counter() - begin)
            model.eval()
            with torch.no_grad():
                begin = time.perf_counter()
                for _ in range(4):
                    model(x)
                inference = (time.perf_counter() - begin) / 4
            row = {'threads': threads, 'input_channels': channels,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'train_batch_seconds': statistics.median(elapsed),
                   'inference_batch_seconds': inference,
                   'rss_GiB': psutil.Process().memory_info().rss / 2**30}
            result['timings'].append(row)
            print(row, flush=True)
    result['limitations'] = 'Synthetic missing Huber/PCC only; excludes real preprocessing, spectral/physiology losses, loading, V0 and checkpoint I/O. No training quality claims.'
    (out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
