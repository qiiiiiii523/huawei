"""Matched B5 continuation: control, frozen S50 and frozen S50-F."""
PROTOCOL = 'B5-S50-three-arm-QC-v2'
SCORE_PROTOCOL = 'correlation50_bonus10_target_record_qc_v3'
FULL_SCORE_PROTOCOL = 'correlation50_bonus10_target_record_v2'
GROUPS = ('control', 'slow', 'slow_f')
SLOW_GROUPS = ('slow', 'slow_f')
INPUTS = ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')
