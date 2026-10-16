"""Deployable B5 fast remainder + predicted S50/S50-F slow curve."""

BUNDLE_PROTOCOL = "B5-best-output-hybrid-v2"
LEGACY_BUNDLE_PROTOCOL = "B5-best-output-hybrid-v1"
SCORE_PROTOCOL = "mean-r-target-record-qc-v1"
SAMPLING = {"seed": 42, "solver": "heun", "steps": 16, "samples": 16}
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
