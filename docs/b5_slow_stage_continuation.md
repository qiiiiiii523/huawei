# S50 / S50-F 慢 CNN 第21～40轮续训

这次继续训练独立慢 CNN，不训练 FM。原 B5 的 Heun16/K16/seed42 预测数组与快残差固定。两组各从自己的原第20轮 `last.pt` 恢复模型、AdamW、随机数和 DataLoader 状态；网络、50秒输入、中间10秒输出、原损失/F系数、学习率、batch、验证间隔均沿用原设置。

## 1. 更新各自服务器 worktree

本地两分支提交后分别推送，服务器在对应 worktree 更新。不要在正在训练的 worktree 切换分支。沿用原实验文档的服务器路径 `/home/qht/huawei_exp_S50` 和 `/home/qht/huawei_exp_S50_F`；如果实际 worktree 名字不同，替换 `cd` 路径即可。

S50：

```bash
cd /home/qht/huawei_exp_S50
git fetch gitee B5-exp-S50
git merge --ff-only FETCH_HEAD
conda activate BioFlow
bash -n scripts/run_b5_slow_continue.sh
```

S50-F：

```bash
cd /home/qht/huawei_exp_S50_F
git fetch gitee B5-exp-S50-F
git merge --ff-only FETCH_HEAD
conda activate BioFlow
bash -n scripts/run_b5_slow_continue.sh
```

若 worktree 是 detached HEAD，`git merge --ff-only FETCH_HEAD` 也可前进到对应分支的新提交；本地有修改时先自行保存，勿强制覆盖。

## 2. 两终端并行启动

终端A，S50，示例GPU0：

```bash
cd /home/qht/huawei_exp_S50
conda activate BioFlow
bash scripts/run_b5_slow_continue.sh --gpu 0
```

终端B，S50-F，示例GPU1：

```bash
cd /home/qht/huawei_exp_S50_F
conda activate BioFlow
bash scripts/run_b5_slow_continue.sh --gpu 1
```

选择空闲GPU，避免与 B5-scratch 等正在运行的任务占用同一张卡。脚本依分支的 C/F 配置选择下表默认路径，不联网、不切分支、不修改原结果目录。

| 项目 | S50 | S50-F |
|---|---|---|
| 旧源目录 | `/home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints` | `/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints` |
| 新阶段根目录 | `/home/qht/huawei/results/B5-exp/slow_continue_r_v1/S50` | `/home/qht/huawei/results/B5-exp/slow_continue_r_v1/S50_F` |

共同 prepared：`/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared`。路径不同时，脚本提供 `--source-run /实际旧checkpoints目录 --root /新的阶段根目录 --prepared /实际prepared目录`。旧目录必须保留原路径和原文件，尤其 S50-F 锁定的原 S50 对照，不要直接覆盖或移动。

`--total-epochs 40` 表示原20轮加新20轮，不是额外40轮。这里的轮次是原 S50/S50-F 实验的轮次，未包含此前 S30 的训练轮次。首次运行不要加 `--resume`。

## 3. 初始候选、评分与保存

开始时先审核旧目录的配置、历史、完整状态、模型/损失代码和既有配对引用，再把现存 `best_slow_r.pt`、`best_score.pt`、`last.pt` 用同一目标导联 QC 重评。若参数相同，只生成一次。旧的60分制总分不进入新阶段选模。

新旧候选统一比较：原 B5 快残差＋当前候选预测慢曲线 → 120秒完整输出 → 相同 QC、目标记录等权、Task2 两设备等权 → `(Task1 r + Task2 r)/2`。

每5轮及最后一轮验证，最终只保存：

| 文件 | 用途 |
|---|---|
| `run/checkpoints/best_r.pt` | 现存旧候选和新阶段已验证候选中，组合平均 r 最高者；可能仍来自旧第20轮或更早 |
| `run/checkpoints/last.pt` | 新阶段最新完整训练状态，供中断恢复 |
| `run/history.jsonl` | 第20轮阶段起点，以及第21～40轮训练、验证和最佳记录 |
| `run/source_candidates_qc.json` | 旧候选在新规则下的评分与哈希 |
| `run/stage.json` | 原目录、数据、实现、预算和来源哈希 |
| `run/completed.json`、`run/report.md` | 完成标志、新旧候选最佳 r1/r2/平均 r 对比 |

只保留这些候选不能找回原先已丢失轮次的权重。慢 r、RMSE、各导联诊断继续记录；它们不决定 `best_r.pt`。训练损失保持原方案，r-only 指的是选模目标。验证指标不能保证新患者成绩同步提升。

## 4. 自动评估与中断恢复

脚本训练结束后自动评估 `best_r.pt`、`last.pt`，输出到 `run/evaluations/best_r` 和 `run/evaluations/last`，包含原始组合预测、逐导联与快慢误差诊断、固定波峰/40秒边界细节图。没有使用真实目标均值校正或真实慢曲线替换。

同一阶段中断后，在原分支、原配置、原新目录运行：

```bash
bash scripts/run_b5_slow_continue.sh --gpu 0 --resume
```

`--resume` 只恢复这个新阶段，不改变已锁定预算；S50-F 把GPU参数换为空闲卡。若仅最终评估中断，同一命令会核验并跳过已完成训练，再恢复评估。不要重跑旧20轮脚本，也不要给旧 `train_loss` 入口直接传40轮。

## 5. 查看与后续使用

两组完成后查看各自 `run/completed.json` 中 `best_score.main_correlation`，以及最终 `run/evaluations/best_r/correlation_score.csv` 的 r1/r2。只有超过按同口径重评的旧候选，才替换 B5-best 的慢预测器候选。新 `best_r.pt` 保留原 SlowCNN 参数命名和推理配置，可供 B5-best 的候选打包入口加载；稳定模型的权重哈希不会自动更新，需另行打包并复核。

打包报告（不含 `.pt`、`.npy`）：

```bash
cd /home/qht/huawei
find results/B5-exp/slow_continue_r_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.log' -o -name '*.png' \) \
  -print0 | tar --null -T - -czf B5_slow_continue_reports.tar.gz
```
