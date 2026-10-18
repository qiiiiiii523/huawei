"""Frozen best slow CNN plus a target-free multiscale residual correction."""
PROTOCOL = 'B5-slow-ms-residual-paired-v1'
SLOW_PROTOCOL = 'B5-slow-ms-residual-v1'
GROUPS = ('control', 'residual')
CORRECTION = {'channels': 24, 'dilations': [[1, 2, 4, 8], [4, 8, 16, 32], [16, 32, 64, 128]]}
