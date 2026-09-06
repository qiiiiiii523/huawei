# GitHub upload

The branch is `baseline/B1`, based on main commit `a626054`. The latest B1 handoff commits are `6718bfd`, `b2c10c4`, and `8d70577`.

The checked branch diff contains no data, checkpoint, prediction, or log artifacts. A portable bundle is saved beside the repository as `D:\\Competition\\ECG\\b1_github_upload.bundle`.

After configuring a GitHub account with write access, run from the B1 checkout:

```powershell
git switch baseline/B1
git push origin baseline/B1
```

If HTTPS credentials are unavailable, add a write-enabled SSH remote and push the same branch. Do not push `main`; open a pull request from `baseline/B1` only after the team reviews the report.
