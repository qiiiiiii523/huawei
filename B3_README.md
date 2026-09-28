# B3 baseline (P0 + P1-C3)

B3 preserves the main `strict_anchor_pretrain`, `joint_anchor_adaptation`,
`P0_anchor_only`, `P1_joint_anchor`, `anchor_i_ecg`, `Y_12lead`,
`same_subject_cross_time`, `same_record_same_window`, and
`joint_anchor_test_like` contracts.

P0 is train-only strict machine-I to d12 reconstruction. This B3 branch runs
only `P0_anchor_only` and `P1-C3`; P1-C3 loads a compatible P0 checkpoint and
uses `film_gated_residual` (FiLM followed by gated residual). The underlying
model still contains the shared four-mode implementation for compatibility,
but B3's training CLI and experiment configs intentionally expose only P0 and
C3. Task 1 uses watch context. Task 2 uses exactly one of
`ecg_machine_d6` or `body_scale_d6` per run; body and machine d6 arrays are
never concatenated.

```powershell
python scripts/train_b3.py --task-id task1 --stage P0_anchor_only --fusion-mode none --output-dir results/b3_task1_p0
python scripts/train_b3.py --task-id task2 --stage P0_anchor_only --fusion-mode none --output-dir results/b3_task2_p0
python scripts/train_b3.py --task-id task1 --stage P1-C3 --fusion-mode film_gated_residual --p0-checkpoint results/b3_task1_p0/b3_best.pt --context-source-type watch_ecg --output-dir results/b3_task1_c3
python scripts/train_b3.py --task-id task2 --stage P1-C3 --fusion-mode film_gated_residual --p0-checkpoint results/b3_task2_p0/b3_best.pt --context-source-type body_scale_d6 --output-dir results/b3_task2_c3
```

Formal inference requires an explicit machine-I anchor and has no target
argument:

```powershell
python scripts/predict_b3.py --checkpoint results/b3_task1_c3/b3_best.pt --task-id task1 --anchor-npy test_machine_i.npy --watch-npy test_watch.npy --output-dir results/test_task1
python scripts/predict_b3.py --checkpoint results/b3_task2_c3/b3_best.pt --task-id task2 --anchor-npy test_machine_i.npy --d6-npy test_body_d6.npy --context-source-type body_scale_d6 --output-dir results/test_task2
```

Validation writes `prediction_raw.npy` and `prediction_submit.npy`; only the
second output replaces lead I with the supplied raw anchor. B3 validation
records raw V0 `r_raw_12`, `r_submit_12`, `r_missing11`, RMSE, and the Task 2
device/subject/V1--V6 diagnostics, including a shuffled-context comparison.

Run the synthetic B3 check with:

```powershell
python scripts/check_b3.py
```

Each checkpoint records `architecture_id` and `architecture_config_hash`.
Missing or incompatible metadata is rejected explicitly, so old M1-era
checkpoints remain available as legacy artifacts but cannot be silently loaded
as B3 P0/P1-C3 checkpoints. Context modules are not considered effective by
this code change; that requires the prescribed validation and shuffled-context
experiments.
