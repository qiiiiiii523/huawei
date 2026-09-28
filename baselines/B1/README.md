# B1 Residual-Dilated U-Net

This directory contains B1-specific protocol notes plus compact historical runners. The current runnable, main-integrated entry point is `scripts/train_b1.py`. Raw ECG arrays, checkpoints, predictions, and logs are intentionally excluded from Git.

## Protocol

- P0: strict same-window `machine I(C) -> machine d12(C)`.
- P1-C3: load the compatible P0 checkpoint, encode cross-time context, then apply FiLM followed by gated residual conditioning.
- Task 1 context: watch I(A).
- Task 2 contexts are mutually exclusive: machine d6(B), body-scale A raw window, and body-scale B after 0.2 Hz detrending.
- Training keeps full d12 supervision and does not replace I. E2 replacement of I is evaluation-only.
- Device-QC `target_quality_mask` is mandatory for the shared training losses and is retained for clean/warning validation diagnostics.
- Checkpoints maximize raw-uV `r_missing11` (II--V6); `r_submit_12` is diagnostic only.
- Seed 42, 500 Hz, 10 s/5000 samples, canonical lead order, frozen train-fitted preprocessing.

## Running

The current protocol-native entry point is `scripts/train_b1.py`, which reads the shared main datasets and device-QC sidecar. The compact `formal_p0.py` and `formal_p1_c3.py` runners are retained only to reproduce historical external-array experiments; they do not replace the protocol-native quality-mask path. Keep data and `--output-dir` results outside the repository.

The scripts use the latest scale-aware physiology loss: frozen d12 scales restore morphology to μV, residual window offsets are removed, and residuals are normalized by the corresponding lead scale.
