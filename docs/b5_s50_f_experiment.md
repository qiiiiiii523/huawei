# S50-F：只增加慢CNN振荡残差监督

分支`B5-exp-S50-F`从已提交的`B5-exp-S50`（3d503ce）创建，使用独立worktree。保留原S50、C/M/F以及正在开发的B5-S50 FM分支，不修改其文件或结果。本轮只训练慢CNN，不训练FM。

## 匹配实验

| 设置 | 已有S50对照 | 新S50-F |
|---|---|---|
| 初始化 | 原S30第50轮best.pt | 同一文件，不使用S50第20轮权重 |
| 输入/输出 | 50秒同步I，中心10秒，拼120秒 | 相同 |
| 网络/人口学/尺度/QC | 已有48通道CNN，213511参数 | 不变 |
| 损失 | Huber＋0.1全记录PCC | 原损失＋固定F项 |
| 预算 | 新增20轮，seed42，每5轮验证 | 相同 |
| 优化器 | 新AdamW，lr1e-4，wd0.01，record batch2 | 相同 |
| 评估 | 固定原B5快分量＋预测慢曲线，60分制v2 | 相同 |

F项是50Hz完整120秒慢曲线上的目标匹配残差Huber：`H(z)=z-居中51点移动平均(z)`，比较`H(预测)`与`H(标签)`。不去掉真实基线漂移，不把振荡全部压为零。F系数固定为已有实验使用的`2.5890171538829843`，不根据验证成绩调整，不重新搜索系数；均值辅助系数始终为0。

预检只在前两条训练记录上报告损失及预测曲线梯度量级。S50输入下的梯度比例可能与旧S30不同，预检不会为了追求某个比例自动改权重。F只用于训练，测试仍只输入I与人口学。

## 服务器操作

1. 本地提交后，将新分支推送到服务器能访问的同一个远程仓库。如果本地使用origin：

```powershell
git -C C:\Users\Ashley\Desktop\HW\huawei push origin B5-exp-S50-F
```

2. 服务器首次创建独立worktree。假设可访问远程名为gitee；实际用origin时替换gitee。

```bash
conda activate BioFlow
cd /home/qht/huawei
git fetch gitee
git worktree add --track -b B5-exp-S50-F /home/qht/huawei_exp_S50_F gitee/B5-exp-S50-F
```

工作树已存在时，进入`/home/qht/huawei_exp_S50_F`执行`git pull --ff-only`，不要重复创建或切换正在训练的主目录分支。

3. 一条命令依次完成无训练预检、S50-F训练、三份候选评估、与已有S50比较。默认GPU2，一张卡。

```bash
cd /home/qht/huawei_exp_S50_F
bash scripts/run_b5_s50_f.sh --gpu 2
```

默认读取：

```text
初始化：/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt
prepared：/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
已有S50对照：/home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints
新结果：/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/
          shared/preflight.json
          S50_F/train.log
          S50_F/checkpoints/best_slow_r.pt
          S50_F/checkpoints/best_score.pt
          S50_F/checkpoints/last.pt
          S50_F/checkpoints/evaluation_{best_slow_r,best_score,last}/
          comparison/report.md
```

不用重新prepare，也不用再训练S50对照。原S50 checkpoint和评估报告必须保留在原位置供校验。原S50现有35.356分只是参考，最终以比较报告记录的匹配候选为准。

## 预检与训练保护

- 预检确认已有S50是同一原S30-best50初始化、seed42、相同20轮预算，且网络/输入/损失定义/评分代码与源版本匹配。
- 不读取S50训练后的模型参数来初始化F，防止变成额外多训练20轮。
- 记录旧S50配置、历史、评估、权重的SHA；新运行前、续跑和比较时检查没有变化。不复制或覆盖原checkpoint。
- 新F第0轮完成实际验证后，要求慢r及原始r1/r2、RMSE/总分与原S50第0轮一致（容差1e-6），不一致则在第一次优化器更新前停止。
- 旧S50是原损失，当前分支默认group=F；日志还应显示`context_seconds=50`。不要用旧`run_b5_s50.sh`启动本实验。
- 预检通过只表示接口、量级及配对一致，不是性能提高的证明。

初始化统一从原S30第50轮加载，但S50-F预算编号仍为新增0～20；0是共同初始化候选。若best为0，说明该指标下新增训练没有超过共同初始化。保留best_slow_r、best_score与last，不能因为训练loss下降就默认使用last。

如需单独预检（不训练），执行：

```bash
cd /home/qht/huawei_exp_S50_F
python -m baselines.B5_exp.check_s50_f \
  --prepared /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared \
  --init-checkpoint /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt \
  --control-run /home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints \
  --report /home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/shared/preflight.json \
  --device cpu
```

确认`ready=true`、`optimizer_updates=0`、`prediction_shape=[2,11,6000]`、`fixed_F_weight=2.5890171538829843`。报告存在时复用，不重复生成。随后直接运行第3步脚本。

## 续跑与结果比较

保持终端连接或在tmux运行；中断后同一工作树执行：

```bash
cd /home/qht/huawei_exp_S50_F
bash scripts/run_b5_s50_f.sh --gpu 2 --resume
```

恢复最新完整epoch，不改预算/初始化/损失。脚本会保留不完整评估再重算，已完成的评估需与当前权重SHA一致。

独立生成比较报告的命令（输出目录应为空）：

```bash
python -m baselines.B5_exp.compare_s50_f \
  --control /home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints \
  --experiment /home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints \
  --output /home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/comparison
```

先比较同等20轮last，再看同一选择规则的best。除总分外，同时检查正常导联慢r/RMSE、两设备胸导联RMSE、独有目标、V2/V3/V4局部波形和拼接边界。收益有限或损害重要指标时保留原S50，不预设组合必然更好。

本轮仍是输出组合；若S50-F确有收益，可把它作为后续慢条件分支权重候选，接入FM方式与S50相同，仍需FM配对实验验证。不会因修改慢CNN训练损失自动改变FM。

## 打包报告

```bash
cd /home/qht/huawei/results/B5-exp
find slow_context_s50_f_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' -o -name '*.png' \) \
  -print0 > /tmp/b5_s50_f_reports.list
tar --null -T /tmp/b5_s50_f_reports.list -czf /home/qht/B5_S50_F_reports.tar.gz
```

下载`/home/qht/B5_S50_F_reports.tar.gz`。不用打包权重和大数组；已有S50对照报告可沿用，比较目录会记录双方成绩。

## 本地检查

35项CPU合成检查通过，覆盖新配对审计、源S30初始化、F系数与预算保护、第0轮分数一致性、候选保存与精确续跑、候选评估与报告比较，以及原S10/S30/S50和B5评分规则。Bash语法及git diff格式检查通过。

测试显式禁止AdamW.step，训练迭代器在第一个训练batch前停止，没有启动真实训练。工作流里的20轮完成元数据为合成测试夹具，用来验证比较接口，不是实验成绩；实际收益需服务器运行。本轮没有推送远程。
