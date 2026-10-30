# B5-final + PCC500 slow-trend + context

This experiment keeps the B5-final fast branch frozen, adds the zero-initialized
PCC500 low-frequency correction to the SlowCNN branch, applies the existing
target-free B5-final context adapter, and rebuilds III/aVR/aVL/aVF from final I
and II.

Validation protocol:

- subject-disjoint official Huawei split;
- 156 subjects with usable training pairs and 40 with usable validation pairs;
- alignment by `task_id + pair_id + start_sample_500hz`;
- baseline R: 0.7071060806;
- PCC500 slow correction: 0.7162049231;
- PCC500 plus B5-final context: 0.7196107731.

The 0.71961 score is a validation result. The saved checkpoint is a B5-specific
PCC500 correction head; it must be used with the matching B5 bundle, branch
scales, and context assets.
