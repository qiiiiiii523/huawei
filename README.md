# 12-lead ECG generation — D0 / D1 / V0 baseline infrastructure

This repository contains the competition data-contract, supervision-control, and evaluation infrastructure. It deliberately contains **no B0/B1/B2 model and no training entry point**. Existing team outputs are read from paths in `configs/common.yaml`; raw recordings are never modified or reparsed.

## D0 — unified data contract

`ecg12gen.UnifiedECGDataset` is the shared reader for task 1 and task 2. An `ECGSample` contains `X_ecg`, `lead_mask`, `Y_12lead`, `missing_mask`, `task_id`, optional `ppg`/`acc`, `meta`, `modality_mask`, and `split`.

- ECG: `uV`, 500 Hz, 10 seconds / 5000 samples.
- d12 order: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
- Task 1: `[N,1,5000] -> [N,12,5000]`; task 2: `[N,6,5000] -> [N,12,5000]`.
- PPG/acc remain optional native-100-Hz fields; v1 does not resample them.
- Dynamic d12 pretraining creates no files: `Y_12lead[:1]` for d12-I and `Y_12lead[:6]` for d12 six-lead.

## D1 — two supervision types, kept separate

`d12_i_pretrain` and `d12_six_pretrain` are synchronous within-d12 reconstruction and are blocked unless `split=train`. Default `cross_device_weak_adaptation` only accepts pair-manifest rows with `pair_status=paired` and usable input/target quality; `review` and `unmatched` cannot automatically enter training.

Every cross-device item records `supervision_mode`, `pairing_type`, `alignment_mode`, `pair_confidence`, and `pair_status`. It is marked `weak_subject_pair_record_start` and `pointwise_mse_allowed=False`, so same-subject pairing is not misrepresented as time-synchronous waveform MSE.

## V0 — one validation protocol

`python -m ecg12gen.evaluate` reports per-lead Pearson r/RMSE, 12-lead mean Pearson/RMSE, r1 (task 1), r2 (task 2), and task-2 missing-lead (V1–V6) mean RMSE. r1/r2 are the unweighted mean of the 12 per-lead correlations. It writes `overall_metrics.csv`, `lead_metrics.csv`, and `report.md`.

```powershell
python -m ecg12gen.evaluate `
  --prediction outputs/task1_validation_prediction.npy `
  --target ../task1_output/task1_validation_target.npy `
  --metadata ../task1_output/task1_window_metadata.csv `
  --task-id task1 --output-dir evaluation_results/task1
```

Use only the fixed `ecg_project/metadata/subject_split.csv`; the reader checks each selected window against it and never randomly splits windows.

## Read-only verification

```powershell
python -m pip install -r requirements.txt
python scripts/check_d0_d1_v0.py --config configs/common.yaml
```

The check memory-maps existing NPY files, validates shapes, lead order, dynamic slicing, subject-level split integrity, D1 gates, and a synthetic V0 report. It does not train a model or use a competition validation target as prediction.