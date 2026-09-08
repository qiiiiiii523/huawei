# M1 final model

M1 preserves the main `strict_anchor_pretrain`, `joint_anchor_adaptation`,
`P0_anchor_only`, `P1_joint_anchor`, `anchor_i_ecg`, `Y_12lead`,
`same_subject_cross_time`, `same_record_same_window`, and
`joint_anchor_test_like` contracts.

P0 is train-only strict machine-I to d12 reconstruction. P1 loads the P0
checkpoint and conditions the same anchor backbone on one global context
embedding. The available independent fusion ablations are `none`, `film`,
`gated_residual`, and `film_gated_residual`. Task 1 uses watch context. Task 2
uses exactly one of `ecg_machine_d6` or `body_scale_d6` per run; body and
machine d6 arrays are never concatenated.

```powershell
python scripts/train_m1.py --task-id task1 --stage P0_anchor_only --fusion-mode none --output-dir results/m1_task1_p0
python scripts/train_m1.py --task-id task1 --stage P1_joint_anchor --fusion-mode film --p0-checkpoint results/m1_task1_p0/m1_best.pt --context-source-type watch_ecg --output-dir results/m1_task1_film
python scripts/train_m1.py --task-id task2 --stage P1_joint_anchor --fusion-mode gated_residual --p0-checkpoint results/m1_task2_p0/m1_best.pt --context-source-type body_scale_d6 --output-dir results/m1_task2_body
```

Formal inference requires an explicit machine-I anchor and has no target
argument:

```powershell
python scripts/predict_m1.py --checkpoint results/m1_task1_film/m1_best.pt --task-id task1 --anchor-npy test_machine_i.npy --watch-npy test_watch.npy --output-dir results/test_task1
python scripts/predict_m1.py --checkpoint results/m1_task2_body/m1_best.pt --task-id task2 --anchor-npy test_machine_i.npy --d6-npy test_body_d6.npy --context-source-type body_scale_d6 --output-dir results/test_task2
```

Validation writes `prediction_raw.npy` and `prediction_submit.npy`; only the
second output replaces lead I with the supplied raw anchor. M1 validation
records raw V0 `r_raw_12`, `r_submit_12`, `r_missing11`, RMSE, and the Task 2
device/subject/V1--V6 diagnostics, including a shuffled-context comparison.

Run the synthetic M1 check with:

```powershell
python scripts/check_m1.py
```
