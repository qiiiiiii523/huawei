# P-prime implementation audit (2026-10-25)

All four independent default configurations implement the selected P-only experiment:

| Branch | Head | Extra endpoint LF weight | Parameters |
|---|---|---:|---:|
| B7-P-L | original linear | 0 | 11477663 |
| B7-P-LF | original linear | 1 | 11477663 |
| B7-P-MS-LF | fine + coarse interpolated velocity | 1 | 11564198 |
| B7-P-LH | nonlinear II/chest heads | 0 | 11520463 |

- Updated M2 base: 0f19a6e, including QC-v2 ff91627. Original M2 untouched.
- All: random initialization, real10s, 500Hz, 5000 points, batch64, 20000 updates,
  lr1e-4, checkpoint500, final-only EMA Heun16/K16, PCC11 coefficient.1;
  FM/MAE II weight5. Dedicated pretrain entry point does not launch A/Q/B.
- QC reader/preparer, noise/ODE, physical limb restoration, evaluator are unchanged
  from the updated M2. Invalid leads are masked in FM, MAE, PCC and LF supervision.
- LF: zero-DC DCT <=.5Hz on normalized endpoint error; no input/output resampling,
  no label replacement at inference, no extra training ODE. Valid time segments
  are projected separately; invalid NaNs cannot contaminate adjacent valid data.
- MS: 21 control points for10s, original full-resolution velocity retained.
  Native CUDA linear interpolation can fail with strict deterministic training.
  Replaced pooling/interpolation by mathematically equivalent cached fixed-matrix
  operations; both forward values and backward gradients independently checked.
  No learned parameters added by this correction, and deterministic mode retained.
- LH: genuinely nonlinear private head processing; seven-channel joint velocity
  and direct clean I preserved. Chest head never consumes a completed II prediction.
- Checkpoint roundtrip/export reconstruction and exact interruption recovery checked.

Verification: 106 tests per branch, 105 pass on CPU, one CUDA-only strict-mode
regression skipped because this local runtime is CPU-only. Includes full10s K16
scoring with a small model and whole-run resume checks. Actual M2-width5000-point
forward/backward smoke tests also passed. This is not a server CUDA launch, GPU
memory measurement or evidence of improved validation scores. The .5Hz cutoff
and LF weight1 are recorded experiment settings, not tuned optima.

Run the optional GPU check on the server from the corresponding checkout with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest baselines.B7.tests.test_multiscale_ops.MultiscaleOpsTests.test_CUDA_strict_deterministic_full_MS_backward -v
```

It uses a small test model and does not read or modify production checkpoints.
Full training still requires the prepared corrected public10 manifest. Do not pull
changed code into a checkout with active training, since resume verifies code identity.
