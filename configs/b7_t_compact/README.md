# B7-T compact candidate recipes

Use `python -m baselines.B7.compact --preset g6_19m ...` or `--preset w144_g5_21m ...`.

Each folder contains all three stage configs. P uses batch64, 10000 updates,
final-only validation (no initial evaluation). Q/A/B use batch32 and 1000 updates.
All stages checkpoint every500. Both candidates start their own P from scratch.

Do not replace `configs/b7_*.json` or baseline checkpoint files. Server worktree,
GPU probe, cache reuse and restart commands: [guide](../../docs/b7_t_compact.md).
