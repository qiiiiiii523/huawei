# B1 Residual-Dilated U-Net

This directory contains only the latest runnable B1 code. Raw ECG arrays, checkpoints, predictions, and logs are intentionally excluded from Git.

## Protocol

- P0: strict same-window `machine I(C) -> machine d12(C)`.
- P1-C3: load the compatible P0 checkpoint, encode cross-time context, then apply FiLM followed by gated residual conditioning.
- Task 1 context: watch I(A).
- Task 2 contexts are mutually exclusive: machine d6(B), body-scale A raw window, and body-scale B after 0.2 Hz detrending.
- Training keeps full d12 supervision and does not replace I. E2 replacement of I is evaluation-only.
- Seed 42, 500 Hz, 10 s/5000 samples, canonical lead order, frozen train-fitted preprocessing.

## Running

Keep data outside the repository. `formal_p0.py` accepts `--data-dir` pointing to a directory containing `strict_train_target.npy` and `strict_validation_target.npy`. `formal_p1_c3.py` accepts `--context-root` pointing to the external context arrays and metadata. Write `--out` to an ignored results directory.

The scripts use the latest scale-aware physiology loss: frozen d12 scales restore morphology to μV, residual window offsets are removed, and residuals are normalized by the corresponding lead scale.
