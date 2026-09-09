# B1 latest loss-v2 run (compact summary)

These are validation results only; checkpoints and prediction arrays remain on the private experiment machine.

| experiment | best epoch | r_submit_12 | r_missing11 | raw RMSE (μV) |
|---|---:|---:|---:|---:|
| P0 | 5 | 0.5375 | 0.4955 | 403.39 |
| Task1 P1-C3 watch | 4 | 0.5213 | 0.4777 | 405.59 |
| Task2 P1-C3 machine | 4 | 0.5247 | 0.4815 | 411.58 |
| Task2 P1-C3 body A raw | 1 | 0.6254 | 0.5914 | 349.66 |
| Task2 P1-C3 body B detrended | 1 | 0.6255 | 0.5915 | 349.63 |

Body A/B used 60 validation windows and must not be compared directly with the 240-window P0 score. Shuffled-context diagnostics are still required before claiming that context is causally useful.
