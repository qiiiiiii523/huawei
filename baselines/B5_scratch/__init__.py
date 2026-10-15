"""B5-scratch: public-pretrained FM, slow condition from Huawei epoch one; not random scratch."""
PROTOCOL = "B5-scratch-early-public-v1"
SCORE_PROTOCOL = "mean-r-target-record-qc-v1"
FULL_SCORE_PROTOCOL = "mean-r-target-record-full-v1"
GROUPS = ("control", "slow", "slow_f")
SLOW_GROUPS = ("slow", "slow_f")
INPUTS = ("anchor", "numeric", "sex", "field_mask", "age_topcoded")
