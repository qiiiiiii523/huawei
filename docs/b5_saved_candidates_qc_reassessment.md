# B5已有候选、S50/S50-F输出组合：统一QC重评

这一步不训练，不改网络，不覆盖原checkpoint或旧评估。使用同一批raw μV预测，同时报告完整集合v2与目标导联QC v3；均按120秒拼接、目标等权、Task2设备等权、50分相关性＋10分RMSE加分计算。QC来自当前main缓存使用的设备QC文件，不根据预测分数过滤。

## 现有B5能做什么，新增了什么

原有 `python -m baselines.B5.validate` 已能加载兼容的旧B5-U checkpoint重新生成并按最新QC评分，默认Heun16/K16。新增 `baselines.B5.reassess_qc` 用来：

- 统一检查原run的best、best_r、best_bonus、last；默认使用原run的resolved_config与实际数据根目录。
- 优先复用有来源证明的旧K16预测，仅在缺少可验证预测且显式允许时重新生成。重新生成只是评估，不构建优化器。
- 给每份原始预测同时算完整集合分和QC分，不改变波形。
- 直接使用S50/S50-F原评估导出的 `learned_slow_hybrid/*/prediction_uV.npy` 重算，无须再训练或再运行慢CNN。
- 输出对比表和现存四份B5候选的QC推荐；不会把输出组合成绩当作FM接入成绩，也不会覆盖best.pt。

## 1. 本地提交后上传B5分支

新增代码在独立worktree：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\QC-rescore`，分支是 **B5**，不在B5-S50或B5-exp-S50-F。

本机手动推送：

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\QC-rescore'
git push origin B5
```

服务器使用BioFlow环境。为避免影响其他正在运行的分支，可单独创建代码worktree；下面远程名gitee按服务器已配置名称使用，需确保该远程已经包含本次B5提交（本地origin与服务器gitee指向不同仓库时，先同步或使用包含新提交的远程）：

```bash
cd /home/qht/huawei
git fetch gitee B5
git worktree add --detach /home/qht/huawei_B5_qc gitee/B5
cd /home/qht/huawei_B5_qc
conda activate BioFlow
sed -i 's/\r$//' scripts/run_b5_qc_reassessment.sh
bash -n scripts/run_b5_qc_reassessment.sh
```

worktree已经存在时进入该目录查看并更新到所需提交，不重复创建；不要在有进程使用的代码目录切分支。脚本内部不fetch、不训练。

## 2. 先完成原B5四份候选重评，GPU1备用

```bash
cd /home/qht/huawei_B5_qc
bash scripts/run_b5_qc_reassessment.sh --gpu 1
```

默认读取：

```text
原B5 run：
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239
    best.pt / best_r.pt / best_bonus.pt / last.pt
    resolved_config.json

旧预测来源证明：
/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
    prepared.json（绑定源best.pt与原K16评估数组的SHA）

新输出：原B5 run/qc_reassessment_v1/
```

程序会先CPU核验文件、读取当前验证目标与质量mask。这时GPU没动是正常的。原prepared已证明的best预测可以直接重算；相同EMA且相同尺度的候选也能复用。其他旧评估若只有文件夹名或checkpoint路径、没有权重和数组SHA，不能可靠证明没被覆盖，因此会重新生成Heun16/K16，使用GPU1。四份里相同EMA只生成一次。

**只允许CPU重算、不希望产生任何生成计算**时，改为：

```bash
bash scripts/run_b5_qc_reassessment.sh --cpu-only
```

若提示某候选没有可验证预测，这是来源证据不足，不是必须重训。随后用原命令加 `--resume`，允许GPU补评即可：

```bash
bash scripts/run_b5_qc_reassessment.sh --gpu 1 --resume
```

中断后也使用这条续跑命令。只复用已完成且SHA匹配的生成，部分生成目录会保留为interrupted再重算；不恢复任何训练优化器。续跑要求原权重、数据、QC、人口学、配置和实现没有变化；改变了需使用新的 `--output`。

## 3. S50-F完成后，把两个输出组合一并重算

原S50、S50-F各自原评估必须先成功，且保留预测大数组。报告tar.gz不含这些数组，不能用压缩包内几个分数直接计算QC。

```bash
cd /home/qht/huawei_B5_qc
run=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239

bash scripts/run_b5_qc_reassessment.sh \
  --gpu 1 \
  --with-s50 \
  --with-s50-f \
  --output "$run/qc_reassessment_with_slow_v1"
```

它会查找同一run下第一轮重评已经生成且有SHA证明的数组，避免再做四份K16生成。S50及S50-F部分仅CPU重算。

默认慢网络评估目录：

```text
S50：/home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints/evaluation_best_score
S50-F：/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/evaluation_best_score
```

若服务器实际输出目录不同，用重复的 `--slow-evaluation` 显式指定，替代with选项：

```bash
bash scripts/run_b5_qc_reassessment.sh \
  --gpu 1 \
  --slow-evaluation S50=/实际路径/evaluation_best_score \
  --slow-evaluation S50-F=/实际路径/evaluation_best_score \
  --output "$run/qc_reassessment_with_slow_v1"
```

本次只重评所指定的慢模型候选。若还要比较best_slow_r或last，换对应评估目录和新的输出目录；不能称只重评best_score就完成了慢网络所有epoch的QC选模。

## 4. 结果看哪里

- `report.md`、`comparison.csv`：两套评分分开列出；横向比较使用同一score_view。
- `selection.json`：现存四份B5中QC总分、QC主相关性、QC加分各自最优的候选。**不改写旧best.pt，也不能恢复已丢失epoch。**
- `B5/best*/target_lead_qc/task*/lead_metrics.csv`：每导联r、RMSE、有效目标数、排除数。
- `S50*/target_lead_qc/`：固定B5快分量＋预测慢曲线的QC参考。
- `reassessment.json`：权重、预测、目标、窗口元数据SHA和来源；`request.json`记录当前QC与验证缓存SHA。
- `generated/`：仅缺失/不可证实的B5预测会新增大数组；`reference/`为本轮当前验证目标与mask副本，未更改数据缓存。

缓存重算会按完整窗口身份匹配当前验证集，核对每个真实目标采样值。顺序可以不同，但不能缺窗口、混目标、改变配对集合或改μV目标。QC由当前目标record_id决定，重复配对/两个设备沿用同一目标mask，坏导联不计分但不删除整条记录。

旧人口学若与训练时的SHA不同会停止，避免一边复用旧条件预测、一边生成新条件预测混比。本轮假设沿用该run已经确认正确的人口学。重新评估不需要公开数据集，也不需要重训公开模型。

## 5. 打包轻量报告

在服务器执行，路径按本次输出选择：

```bash
cd /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239
find qc_reassessment_with_slow_v1 -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.md' \) \
  -print0 | tar --null -T - -czf /home/qht/B5_QC_reassessment_reports.tar.gz
```

下载 `/home/qht/B5_QC_reassessment_reports.tar.gz`。不含权重或波形大数组。完成这一步后，再根据同一QC口径选择后续FM接入实验的参考；尚未修改B5-S50的评分或启动其训练。

## 本地检查范围

7项新增合成数据检查和19项现有main-v2、选模、采样回归检查通过，Bash语法检查通过。新增检查覆盖raw目标逐点核对、乱序窗口对齐、逐导联QC、旧缓存来源验证、四候选与两个hybrid汇总、相同EMA复用、续跑与变更拒绝，以及补评调用现有EMA验证和固定Heun16/K16。文件流程测试中的生成使用合成替身，明确禁止优化器构建/更新；没有运行真实数据训练或GPU评估。服务器真实文件、显存和实际分数需按上述流程验证。
