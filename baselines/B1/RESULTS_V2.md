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
