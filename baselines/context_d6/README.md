# Task 2 machine-d6 context experiments

This package adapts the frozen B5 checkpoint for the ECG-machine six-lead history condition.
It intentionally does not implement the body-scale branch or Task 1.

Paired experiments:

- `configs/experiments/context_d6_control.yaml`: same machine joint targets, context disabled.
- `configs/experiments/context_d6_context.yaml`: same targets and budget, masked full-record context enabled.

The context encoder receives `[batch, windows, 6, 5000]`. Invalid samples, bad leads, tail padding,
and unavailable windows are zeroed by the shared collator. Window embeddings are pooled using valid-point
weights. Zero-initialized multiscale affine residuals condition the inherited B5 anchor features, so both
groups reproduce B5 exactly before the first optimizer update.

Exploratory configs use Heun4/K1 to keep periodic validation tractable. Final comparison must change both
configs together to Heun16/K16, seed 42, then start fresh output directories.

```powershell
$python = "D:\\path\\to\\python.exe"
& $python -m unittest baselines.context_d6.tests.test_context_d6 -v
& $python -m baselines.context_d6.train --config configs/experiments/context_d6_control.yaml --device cuda --execute-training
& $python -m baselines.context_d6.train --config configs/experiments/context_d6_context.yaml --device cuda --execute-training
```

Do not compare either adapted group to the received B5 score as a substitute for the paired control.

