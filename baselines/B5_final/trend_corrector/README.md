# B5-final 120-second TCN corrector

This module adds a target-free inference corrector and its training tools to
the B5-final pipeline.

The corrector follows the final lead contract exactly:

- input features: B5-final predictions for `I, V1--V6`;
- direct correction outputs: `II, V1--V6`;
- `I` remains the B5-final prediction;
- `III`, `aVR`, `aVL`, and `aVF` are recomputed from the final `I` and `II`;
- every 120-second correction curve is zero mean.

## Workflow

1. Run the existing B5-final validation/study pipeline once to generate the
   120-second final study and baseline evaluation.
2. Freeze prediction-only inputs with `cache_predictions.py`.
3. Train with `train.py`; labels are read from the immutable baseline
   evaluation and are never copied into the prediction cache.
4. Apply the selected checkpoint with `apply.py`.

The trainer selects `best.pt` by centered full-120-second correlation over
held-out subjects.  The cache splits by `subject_id`, so every input form and
every 120-second record from one participant stays on the same side.

Run the unit tests with:

```bash
python -m unittest baselines.B5_final.tests.test_trend_corrector -v
```
