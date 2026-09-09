# B1 protocol-native formal run (main `3ba5eb8`)

RTX 3080 Ti 12 GB, seed 42, fixed subject split, main preprocessing/loss/V0 evaluator, batch 8. Checkpoints use validation `r_submit_12`; E1 is raw model output and E2 replaces I only at evaluation.

| experiment | n(val) | best epoch | E2 r_submit_12 | E2 r_missing11 | E2 raw RMSE (μV) | E2 centered r / RMSE (μV) |
|---|---:|---:|---:|---:|---:|---:|
| P0 Task1 | 213 | 4 | 0.5169 | 0.4730 | 454.85 | 0.7125 / 181.78 |
| P0 Task2 (all) | 300 | 5 | 0.5251 | 0.4819 | 453.08 | 0.7261 / 177.86 |
| P1-C3 Task1 watch | 213 | 1 | 0.5188 | 0.4750 | 454.97 | 0.7121 / 181.60 |
| P1-C3 Task2 machine d6 | 240 | 15 | 0.5181 | 0.4743 | 459.46 | 0.7219 / 182.54 |
| P1-C3 Task2 body d6 raw | 60 | 1 | 0.5723 | 0.5334 | 438.97 | 0.7565 / 169.83 |
| P1-C3 Task2 body d6 detrended | 60 | 1 | 0.5718 | 0.5329 | 439.10 | 0.7560 / 169.95 |

Task-2 V1–V6 E2: machine `r=0.3390`, centered `r=0.6044`, centered RMSE `279.49 μV`; body raw `r=0.4409`, centered `r=0.6846`, centered RMSE `257.74 μV`; body detrended `r=0.4405`, centered `r=0.6842`, centered RMSE `257.91 μV`.

On the same subsets, P0 gives `r_submit=0.5158` (machine, n=240) and `0.5779` (body, n=60). Therefore C3 gain is small for machine (+0.0023) and negative for body; raw/detrended are tied. Shuffle scores were near or above matched scores, so these runs do **not** establish that context is being used.

Older loss-v2 numbers in previous revisions are exploratory and not comparable to this table.

## P0 optimization sweep (main `3ba5eb8`)

All candidates used the same protocol, seed 42, batch 16, 40 epochs, cosine LR (`1e-4`→`5e-6`), and an auxiliary train-only baseline-head loss. Only the B1 backbone variant changed.

| variant | Task1 best epoch / E2 r_submit | Task2 best epoch / E2 r_submit | centered r (T1 / T2) |
|---|---:|---:|---:|
| base | 14 / 0.4931 | 14 / 0.5034 | 0.6831 / 0.6990 |
| wide (32/64/128/256) | 6 / **0.5138** | 6 / **0.5219** | **0.7087 / 0.7235** |
| dilated bottleneck | 9 / 0.5131 | 8 / 0.5185 | 0.7060 / 0.7196 |

The wider U-Net is the current P0 candidate. It did not exceed the earlier protocol-native P0 scores (`0.5169` Task1, `0.5251` Task2), so baseline supervision plus this schedule is not yet a net improvement. Chest leads remain the limiting factor (best wide Task2 E2 r: V1 `.301`, V3 `.207`, V6 `.231`). The next high-value P0 experiment is the constrained seven-output/analytic-limb head, followed by a fair re-evaluation of the best backbone.

## Core7 analytic-limb P0

The head freely predicts `II+V1–V6`; I is the strict anchor identity path, and III/aVR/aVL/aVF are generated in μV morphology space before conversion back to centered-scaled d12. With the same 40-epoch budget and optimizer, the best checkpoints were epoch 3: Task1 E2 `r_submit=0.5122`, `r_missing11=0.4679`, centered `r=0.7035` / RMSE `183.93 μV`; Task2 E2 `r_submit=0.5213`, `r_missing11=0.4778`, centered `r=0.7178` / RMSE `179.86 μV`. Task2 V1–V6 centered `r=0.6181`, RMSE `271.03 μV`.

Core7 therefore did not beat the existing full-12 P0 (`0.5169` / `0.5251`) under this schedule. It remains a useful constrained-output candidate, but should not replace the current P0 checkpoint without a dedicated loss/learning-rate sweep.

## Soft analytic and temporal-attention P0

These architecture-only tests retained the same main data, loss, seed, batch 16, 40-epoch budget, optimizer, and checkpoint rule used by the optimization sweep. `softcore` learns a soft blend between freely predicted and analytically derived limb leads; `attention` adds a two-layer temporal Transformer at the 625-token U-Net bottleneck.

| variant | task | n(val) | best epoch | E1 r_raw_12 | E2 r_submit_12 | r_missing11 | E2 raw RMSE (uV) | E2 centered r / RMSE (uV) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| softcore | Task1 | 213 | 5 | 0.4865 | 0.5128 | 0.4686 | 456.59 | 0.7077 / 183.98 |
| softcore | Task2 | 300 | 5 | 0.4941 | 0.5212 | 0.4777 | 454.76 | 0.7226 / 179.78 |
| attention | Task1 | 213 | 6 | 0.4870 | **0.5131** | 0.4688 | 456.05 | **0.7103 / 182.28** |
| attention | Task2 | 300 | 5 | 0.4917 | 0.5192 | 0.4755 | 455.00 | **0.7231 / 178.56** |

The soft analytic gates stayed close to their 0.20 initialization (`0.2021–0.2026`), so the model did not discover a benefit from stronger analytic enforcement. Attention improved centered diagnostics slightly relative to the wide sweep, but neither candidate exceeded the original protocol-native P0 (`0.5169` Task1, `0.5251` Task2). These variants are retained as negative ablations; the original full-12 P0 remains the production anchor.
