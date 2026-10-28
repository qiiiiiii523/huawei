# C5 + learned mean correction

The frozen-C5, 31,687-parameter one-second mean correction experiment and
target-free inference entry point are documented in
[experiments/c5_mean_correction/README.md](../experiments/c5_mean_correction/README.md).

This branch adds the small residual corrector, not the separate experiment
that fine-tunes the base FM with a final-generated mean loss.
