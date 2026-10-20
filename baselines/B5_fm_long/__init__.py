"""Paired 10s/40s full-wave FM experiment; Final defaults are untouched."""

PROTOCOL = "B5-final-FM40-paired-v1"
PACKAGE_PROTOCOL = "B5-final-FM40-inference-v1"
FIELDS = ("numeric", "sex", "field_mask", "age_topcoded")
WINDOW_SECONDS = (10, 40)
MODES = ("raw", "hybrid")
