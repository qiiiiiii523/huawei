# B7 P-prime: four independent public10 experiments

Current branch: **B7-P-LF**, original linear M2 head, PCC11 plus LF endpoint loss.

All four branches start at the updated M2 `0f19a6e` (includes QC-v2 `ff91627`).
The original M2 checkout/configs/cache/checkpoints are not modified. Only the
dedicated `configs/b7_pprime.json` selects the experiment. Shared implementation
supports all head types so checkpoint reconstruction/export and tests agree.

| Branch | Head | LF coefficient |
|---|---|---:|
| B7-P-L | original linear 112-to-7 | 0 |
| B7-P-LF | original linear 112-to-7 | 1 |
| B7-P-MS-LF | original fine path plus coarse velocity head | 1 |
| B7-P-LH | separate nonlinear II and chest heads | 0 |

Exact parameter counts: L/LF **11,477,663**; MS-LF **11,564,198**;
LH **11,520,463**. These counts were checked with the actual M2 widths, and each
variant passed a CPU5000-point forward/backward smoke test. GPU peak memory and
validation improvement are not established by that smoke test.

All use the same corrected **QC-v2 public10 manifest**, fitted train-only scales,
and clean I. No recaching is required if that shared corrected cache is complete.
Preflight reads cached arrays/shape/masks, not raw WFDB or ODE validation. It reports
progress, rejects old data, wrong length/rate, extra datasets and unreliable I.
No original weights are loaded. New runs initialize randomly; normal interruption
recovery restores this same run's model/EMA/optimizer/RNG/budget from last.pt.

## Common configuration (updated M2)

- Widths 112/224/448; bottleneck dilations 1/2/4/8/16/32; time/metadata 128.
- Real 10s at **500Hz**, 5000 input and output points; no resampling or padding.
- Clean I + seven states = eight input channels; seven output velocities ordered
  II,V1,V2,V3,V4,V5,V6. Fixed physical-unit formulas restore the other limb leads.
- AdamW lr 1e-4, weight decay .01, batch64, **20000 updates**; clip1, EMA .999.
- FM7 + MAE7 (II weight5) + .1 PCC11 (11 missing leads equal weight).
- Full 10s MAE/PCC, mask-aware; I never included as a prediction/score target.
- Save recovery checkpoint every500 updates; no initial/intermediate full
  validation; **final EMA only**. Final evaluation: Heun16, K16 waveform mean,
  sample batch16, record batch4. No best-step search under final-only selection.
- Extra heads are initialized without changing trunk weights or the subsequent
  training RNG state relative to the baseline with the same seed.

## Low-frequency definition (LF and MS-LF)

This is an experimental choice, not a proven optimal weight/cutoff. Add
`1.0 * MAE_weighted(project_slow(endpoint - target))`, with II weight5 and the same
frozen per-lead normalization as ordinary MAE. Endpoint is `x_t+(1-t)*v_theta`.
Keep the full FM, ordinary MAE and PCC11 objectives. No extra ODE rollout or K16
generation occurs during each training update.

`project_slow` is a differentiable **DCT-II orthonormal projection** onto modes
`0 < f <= 0.5Hz`, excluding DC. For 5000 samples at500Hz there are10 modes.
This preserves 500Hz and the original waveform length; it does not downsample
either input or target. It differs from the diagnostic Butterworth filter, while
using the same physical frequency scale. There is no causal-filter requirement
for full-window, non-streaming ECG reconstruction.

Invalid leads/points are excluded BEFORE arithmetic. No filter crosses masked
gaps: each contiguous valid segment is projected separately, using its true
duration. Segments shorter than one second have no non-DC DCT mode at<=.5Hz and
do not contribute. Constant offsets remain constrained by ordinary MAE.

The LF term supervises the **sum** of velocity paths through the endpoint, not
the coarse head alone. The oracle's label replacement is never used in inference.
Training endpoint r is not a substitute for free-generation validation r.

## Structural candidates

LH: each private head is Conv3(112,64), SiLU, Conv1(64,1 or6). This adds nonlinear
lead-specific processing; merely splitting the original linear convolution
would have been equivalent. No II-to-chest cascade or second FM is introduced.

MS-LF: time/metadata-conditioned bottleneck features are pooled to uniform knots,
then Conv3(448,64), SiLU, Conv1(64,7), interpolated linearly to5000 points and
added to the original fine velocity path. Max physical interval .5s gives21
knots for10s (81 for40s in reusable model code). The last coarse projection is
zero-initialized, so initial predictions match the fine-path baseline. The coarse
head does not replace full-resolution noise removal. Neither path is guaranteed
to become an identifiable physiological frequency band. No attention/two-stage/
inverse-reconstruction module is added.

MS deterministic-training correction:pooling and endpoint-aligned linear
interpolation use fixed matrices and matrix-product backward, instead of native
CUDA adaptive-pool/linear-interpolation backward kernels. This preserves their
mathematical definitions and all learned parameters. Native CUDA linear
interpolation can throw under the trainer's strict deterministic mode; do not
disable that mode to bypass the issue. CPU equivalence tests check values and
gradients; a CUDA-specific regression is provided and skips on CPU-only hosts.

## Server launch (P-prime ONLY)

After pushing the corresponding branch, fetch it into a new server checkout or
worktree. Run the following there; change only the GPU/output path for that run.
The manifest path must point to the already completed shared QC-v2 public10 cache.

```bash
conda activate BioFlow
# cd /home/qht/<checkout-for-this-branch>
mkdir -p results
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B7.pretrain \
  --manifest /home/qht/huawei/data_cache/B7_public_qcv2/data/public10/manifest.json \
  --output results/Pprime_LF_seed42 --device cuda \
  > results/Pprime_LF_seed42.log 2>&1
```

Use output/log suffix `LF`, `MS_LF` or `LH` on its corresponding branch. Run in a
tmux session if unattended. An interrupted run uses exactly the same command
plus `--resume`; do not edit code/config/cache during a run. No `--init-checkpoint`
option exists on this P-only launcher. It does not prepare PTBDB/Huawei and never
launches A/Q/B. Do not use the inherited full `pipeline` launcher for these runs.

Each run owns a new output/log directory. GPU allocation/concurrency is explicit;
these four runs do not provide a cross-process GPU memory scheduler. They keep
500Hz output even in LF/MS modes. New QC-v2 scores are a separate protocol cohort
from historical leaked/invalid-supervision results; old0.74 is not a clean paired
baseline. Report both dataset groups, each lead, valid counts and LF metrics;
select candidates by fixed validation rules and reserve test for final reporting.

## Local verification

CPU regression: `python -m unittest discover -s baselines/B7/tests -t .` (82 tests: 81 CPU passes, one CUDA-only check skipped locally)
and `python -m unittest scripts.test_b7_structures scripts.test_b7_large
scripts.test_recover_b7_ptb scripts.test_reschedule_b7_validation` (24 tests).
Includes real K16 full10s inference/scoring with a small test model, actual-M2
5000-point forward/backward smoke checks, physical frequency/DC/fast-mode checks,
QC masks/NaNs, head order, zero-start MS, checkpoint reload and exact interrupted
recovery. Validation improvement, GPU peak memory and full server training are
not claimed by these local CPU checks.
