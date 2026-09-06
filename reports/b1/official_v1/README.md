# B1 official-v1 handoff

This directory is reserved for the compact, team-facing B1 report. It must contain only small CSV/Markdown summaries and must not contain datasets, checkpoints, prediction arrays, or full training logs.

The official matrix has 14 runs:

- task1: A_L0, A_L1, B_L0, B_L1, C1, C2;
- task2: data A/B × route A/B × L0/L1.

Each run uses the frozen main protocol: seed 42, deterministic execution, batch size 16, 100 epochs per stage, AdamW (learning rate 0.001, weight decay 0.0001), no scheduler, no gradient clipping, no early stopping, and checkpoint selection by validation raw-uV V0 (`r1` for task1 and `r2` for task2).

The full run directory is intentionally outside Git at `results/b1/official_v1/`. The remote AutoDL working copy is `/root/autodl-tmp/huawei_B1_official/`. Run `python -m b1.summarize_official` there after all 14 runs finish, then copy only `delivery/` into this directory.

The implementation commit is `6718bfd` on branch `baseline/B1`. The branch exposes only the official-v1 handoff files; superseded prototype outputs are kept in Git history and are not part of the current tree.
