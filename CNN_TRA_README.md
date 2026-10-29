# CNN+TRA: I -> II and V1--V6

This branch starts from `main` and reuses the B3 10-second CNN pyramid,
4-layer bidirectional time Transformer and FPN decoder. It accepts only the
same-record, same-window lead I. It predicts seven leads in this fixed order:
`II,V1,V2,V3,V4,V5,V6`. No demographics, watch context, d6 context, formula
leads or correction network are used by this stage.

## Data and schedule

- Data: `/root/autodl-tmp/huawei_data`, raw microvolts, 500 Hz.
- Train selection: `metadata/d12_strict_pretrain_index.csv`, with D12 record
  lead-quality masks from `metadata/device_interpretation_qc.csv`.
- Input I marked unreliable: exclude the full record. Unreliable output lead:
  mask only that lead's loss.
- Validation: preassigned task1/task2 validation arrays and quality masks.
- Training: 5000 optimizer updates, validation at steps 3000 and 5000 only.
- Loss: normalized Huber + `0.1 * (1 - PCC)`; lead weights `II=3`, `V1..V6=1`.
- Optimizer defaults follow B3 P0: AdamW, batch size 4, learning rate 1e-3,
  weight decay 1e-4, seed 42.

Run from the repository root:

```bash
/root/miniconda3/bin/python scripts/train_cnn_tra.py \
  --output-dir /root/autodl-tmp/cnn_tra_runs/first_run
```

The `--smoke-test` option performs one real forward/backward update using the
data without creating output files or starting the 5000-step run.

## Validation caches

Each `step_003000/` and `step_005000/` directory contains `task1/` and
`task2/`. Each task cache has `prediction_7_uV.npy`, `target_7_uV.npy`,
`input_i_uV.npy`, 7-lead target QC, input-I QC, `window_metadata.csv` and
`metrics.json`. The arrays preserve the original validation window ordering;
metadata groups every 12 consecutive windows into the full 120 seconds. The
`COMPLETE` file is written last.

The reported score is a **provisional seven-lead** 120-second Pearson-r score,
with RMSE as a secondary diagnostic. Task1 and Task2 contribute equally;
Task2's machine and body-scale groups contribute equally. This is not the
final 11-missing-lead score, which requires the later correction stage.
The cache deliberately contains no reconstructed 12-lead prediction.
