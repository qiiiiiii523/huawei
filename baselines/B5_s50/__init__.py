"""Paired B5 continuation with frozen S50 predictions; never trains on import."""
PROTOCOL = 'B5-S50-frozen-condition-v1'
SCORE_PROTOCOL = 'correlation50_bonus10_target_record_v2'
INPUTS = ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')
