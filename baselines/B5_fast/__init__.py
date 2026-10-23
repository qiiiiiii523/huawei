"""Frozen original B5 FM fast residual; no slow predictor or optimizer."""

PROTOCOL = "B5-fast-frozen-v1"
COMPONENT_PROTOCOL = "B5-slow-additive-uV-v1"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MISSING_LEADS = LEADS[1:]
