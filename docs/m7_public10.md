# Fixed M7 public10 preset

This branch is **B7-P-LF / M7-LF**. All four independent branches share the
same frozen M7 backbone. Only output-head type and extra LF term differ.

| Experiment | Head | Extra LF coefficient | Exact parameters |
|---|---|---:|---:|
| M7-L | linear64-to-7 | 0 | 14502815 |
| M7-LF | linear64-to-7 | 1 | 14502815 |
| M7-MS-LF | fine velocity + coarse-control velocity | 1 | 14552486 |
| M7-LH | private nonlinear II/chest heads | 0 | 14527519 |

## Backbone and public training

- Seven resolution scales: **64/128/256/256/256/256/256** channels, six down/up
  levels with symmetric skips, two encoder/decoder residual blocks per skip scale.
- Deepest bottleneck: six blocks, dilation1/2/4/8/16/32; time/metadata dims128.
- Clean I is channel0, never noised or independently reconstructed. Seven noisy
  states generate seven velocities ordered II,V1-V6; formulas recover12 leads.
- Real10s at500Hz:5000->2500->1250->625->312->156->78, decode to exact5000.
- Same completed QC-v2 public10 manifest, split, masks, frozen train-only scales.
- **Random initialization**, batch64,20000 updates, AdamW lr1e-4, weight decay.01,
  gradient clip1, EMA.999, recovery checkpoint every500 updates, seed42.
- **One final full validation only**, no initial/intermediate full validation;
  final EMA with fixed Heun16, **K16 waveform mean**, ODE batch16, record batch4.
- FM7 + MAE7 (II weight5) + .1 PCC11 (missing11 equal weight).
- LF variants additionally use coefficient1 * zero-DC DCT<=.5Hz endpoint MAE7,
  II weight5, with mask-safe contiguous segments. Data remain at500Hz.
- MS has21 knots at.5s maximum spacing for10s, interpolated to full velocity via
  deterministic fixed matrices and added to full-resolution path. Zero-start
  coarse output. LH uses Conv3(64,64), SiLU, Conv1(64,1 or6).

The canonical configuration is **configs/u7_m7/b7_public10.json**. The old default
path configs/b7_pprime.json remains an identical alias. Both are checked together.
The inherited configs/u7_m2 files are unchanged historical reference settings;
they are not the M7 launcher default. Logs print preset=M7.

## P-prime only launch

From this branch's own server checkout/worktree after push/fetch, in tmux:

```bash
conda activate BioFlow
mkdir -p results
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B7.pretrain \
  --manifest /home/qht/huawei/data_cache/B7_public_qcv2/data/public10/manifest.json \
  --config configs/u7_m7/b7_public10.json \
  --output results/M7_LF_seed42 --device cuda \
  > results/M7_LF_seed42.log 2>&1
```

Choose the GPU explicitly. No GPU probes, VRAM limits or concurrency scheduler
are added here. This P-only launcher does not run A/Q/B, raw preparation or any
old-weight initialization. It checks the corrected cache before training.

To recover **this new M7 run** unchanged, use the same command plus --resume.
Do not resume/initialize from the former40.8M seven-scale or three-scale M2 model:
channel shapes differ. Use a fresh output directory; reuse the same QC-v2 cache.
Parameter budget does not itself guarantee a GPU memory bound or improved r.

## Local verification

All115 previous tests plus5 M7 preset tests are run on each branch:120 total,
118 CPU passes and2 CUDA-only checks skipped on this CPU runtime. Tests enforce
10M-20M parameters, exact head counts, canonical/alias identity, M2-equivalent
training/sampling and P-only CLI default; they retain seven-scale shape/gradient,
checkpoint/export/K16 scoring and exact resume checks. Actual-width M7 full5000-
point forward/backward smoke checks are additional; no server GPU memory or
validation gain is claimed by local checks.
