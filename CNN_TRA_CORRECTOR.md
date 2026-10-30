# Two-stage 120-second corrector for CNN+TRA

Base: frozen CNN+TRA step-3000 checkpoint, whose provisional seven-lead
validation `r` was higher than step 5000. Fit data come only from the strict
training index; the step-3000 validation prediction cache is evaluation-only.

Pipeline:

1. Infer all 156 strict training records with the frozen CNN and cache
   120-second raw-uV I, seven predictions, D12 targets and quality masks.
2. Train a B7-style 0.1-second additive corrector for II,V1..V6 from scratch.
   II receives loss weight 3; each chest lead receives weight 1. Select its
   checkpoint using the current `main` reliable-input, full-120-second,
   missing-11 mean correlation protocol.
3. Freeze the selected new 0.1-second corrector. Train a B7-style chest-only
   4-Hz mid correction for V1..V6 from scratch. Select on the same protocol.
4. Derive III,aVR,aVL,aVF from observed I and corrected II only after both
   learned corrections. The final missing-lead output order is
   II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6.

The B7 weights from `huawei_B7_trend_corrector` are reference architecture
only; neither old weight file is loaded. Targets enter losses and validation,
never corrector features. Demographics and other context are not used.

On the remote machine, from the repository root:

```bash
/root/miniconda3/bin/python scripts/train_cnn_tra_correctors.py \
  --phase all \
  --output-dir /root/autodl-tmp/cnn_tra_runs/corrector_v1
```

Phases `prepare`, `fine`, and `mid` can also run separately. Completed phases
are skipped by `all`; incomplete nonempty phase outputs cause an error so an
interrupted run cannot silently overwrite a previous result.

This run uses the original B7 schedule for each new corrector: 2000 updates,
batch 8, AdamW learning rate 3e-4, weight decay 1e-4, validation every 250
steps including step 0. Metrics are validation results, not an independent
test estimate.

After both phases finish, export the aligned corrected validation cache:

```bash
/root/miniconda3/bin/python scripts/export_cnn_tra_corrected_validation.py
```

`corrector_v1/final_validation/task1/prediction_11_uV.npy` and the matching
`task2` file have shape `[windows, 11, 5000]`; rows match the original
step-3000 validation index. `manifest.json` records the metric and model
hashes. These validation predictions were never used for fitting either
corrector.

To correct one full 120-second CNN prediction, provide raw-uV I `[1,60000]`,
raw-uV CNN output `[7,60000]` in II,V1..V6 order, and the exact frozen CNN
checkpoint:

```bash
/root/miniconda3/bin/python scripts/apply_cnn_tra_correctors.py \
  --input-i /path/to/input_i.npy \
  --base-seven /path/to/prediction_7.npy \
  --cnn-checkpoint /root/autodl-tmp/cnn_tra_runs/first_run/model_step_003000.pt \
  --fine-checkpoint /root/autodl-tmp/cnn_tra_runs/corrector_v1/fine_0p1/best.pt \
  --mid-checkpoint /root/autodl-tmp/cnn_tra_runs/corrector_v1/chest_mid/best.pt \
  --output /path/to/prediction_11.npy
```
