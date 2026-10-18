"""Paired B/SM/L/ALL scale allocation with an explicit common training seed."""
PROTOCOL = "B5-control100-scale-ablation-reliable-I-v3"
LEGACY_CACHE_PROTOCOL = "B5-best-SlowPCC-fixed-FM-records-v2-pair-validation"
GROUPS = ("adapt_control100", "adapt_short_middle", "adapt_long", "adapt_multiscale")
BANDS_BY_GROUP = dict(zip(GROUPS, ((), (0, 1), (2,), (0, 1, 2))))
