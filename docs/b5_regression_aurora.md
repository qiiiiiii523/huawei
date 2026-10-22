# B5 Regression AURORA

This branch is a deterministic, physiology-decomposed regression experiment inspired by AURORA and IGUANA. It is not an official reproduction because the cited repository does not currently contain implementation code.

The model learns Lead I to Lead II, derives the other limb leads with exact Einthoven and Goldberger identities in raw microvolts, and uses a second independent lead-conditioned 1D U-Net to generate V1 through V6 from all six limb leads. Training combines target reconstruction, PCC, reverse source reconstruction, and representation consistency. Cycle branches are disabled during inference.

Public pretraining and Huawei fine-tuning must share one frozen scale artifact. Flow Matching, old direct-regression, and this branch's checkpoints are intentionally incompatible. Final selection uses the existing reliable-input and target-QC record protocol with equal weighting of Task2 devices.

Run scripts/run_b5_regression_aurora.sh with a first argument of prepare, public, huawei, or evaluate after exporting the variables referenced by that action. Use a new empty output directory for every run.

The predeclared gate is: stop this regression direction below 0.68 baseline main correlation; tune consistency objectives only in the 0.68 to 0.70 range; compare deployment speed only after approaching or exceeding F0 baseline.
