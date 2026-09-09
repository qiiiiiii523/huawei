# B1 protocol audit (main 7aa6508)

- Public contract: `configs/context_fusion_protocol.yaml`, `configs/training_protocol_v1.yaml`.
- Strict pretraining index: train-only, deduplicated by target record/window; 962 rows (task1 855, task2 107).
- Subject split: 88 train / 22 validation; seed 42; 500 Hz, 5000 samples, canonical 12-lead order.
- P0: machine I(C) -> machine d12(C), full 12-lead output, no training-time I replacement.
- P1 baseline: same architecture P0 checkpoint -> C3 (FiLM then gated residual), gate 0.05, zero-initialized residual last layer.
- Task 1 context: watch I(A). Task 2 contexts are mutually exclusive machine/holter d6(B) and body-scale d6(A); no P1-both.
- Official checkpoint metric: validation test-like `r_submit_12` / raw-uV V0. Centered metrics are diagnostics only.
- Required reporting: r_raw_12, r_submit_12, r_missing11, RMSE, per-lead; Task2 machine/body, subject-macro, V1-V6 and shuffled-context diagnostics.

## Compatibility findings

1. The local data products have one extra nested directory level (`task1_output/task1_output` and `task2_body_scale_ablation/task2_body_scale_ablation`). They are left untouched; B1 readers must resolve the nested read-only paths.
2. Earlier B1 P0 scripts selected checkpoints by `r_missing11` or used zero baseline and the structured prototype hard-copied I. Those results remain exploratory architecture evidence, but are not formal main-protocol results.
3. The formal rerun must use the frozen preprocessing and `joint_anchor_sync_loss` contract, then attach C3 from the exact P0 architecture/configuration.
