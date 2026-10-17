# B5系列同步：可靠I验证＋两任务平均r

## 已同步范围

- main共享提交：e490d5f。
- 稳定B5、B5-best：均接入实际同步I的可靠性过滤；新运行按R=0.5r1+0.5r2选best_r.pt，RMSE不计分。
- 训练、网络、损失、尺度、人口学、Heun16/K16及β=0.75不变。新训练配置使用带reliable_i_v1的独立输出目录。
- 现有模型包保持字节和推理兼容。旧best账本不能作为新范围的--resume；旧权重仍可用于推理或新运行初始化。
- SlowPCC、V13、S50等已完成的冻结实验分支不直接修改。它们的结果由新的稳定评分环境重算，避免破坏study代码SHA。

## 1. 本机推送两个稳定分支

本机工作树：

```text
B5      C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\QC-rescore
B5-best C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best
```

在任意该仓库工作树执行：

```powershell
git push origin main B5 B5-best
```

服务器使用gitee；执行下面步骤前确认这些提交已同步到gitee对应分支。若本机远程不是同一镜像，应先按既有仓库同步方式将提交送达，不能假定push到origin就一定更新gitee。

## 2. 服务器更新现有B5-best工作树

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best
git status --short
git fetch gitee B5-best
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

存在冲突时保留本地文件，不执行reset --hard，不删除训练结果。不要在其他正在训练的工作树切换分支。

如需B5纯FM的新工作树，首次创建：

```bash
cd /home/qht/huawei
git fetch gitee B5
git worktree add --detach /home/qht/huawei_B5_qc FETCH_HEAD
```

已有该工作树时在其中fetch并merge --ff-only，不重复创建。不需要为了CPU重算同时创建所有实验工作树。

## 3. 先重算当前默认最佳组合，不需要GPU

当前默认模型的原部署评估数组已存在。执行：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best

out=/home/qht/huawei/results/B5-rescore-reliable-I/default_$(date +%Y%m%d_%H%M%S)

python -u scripts/rescore_reliable_i.py \
  --evaluation-dir /home/qht/huawei/results/B5-best-cross/deployment_check_20261017_100949_439149 \
  --input-quality-source same_record_target_I \
  --output-dir "$out"

cat "$out/correlation_score.csv"
echo "新范围结果：$out"
)
```

这项是当前control100＋SlowPCC20默认包的重评分。它不是旧原FM＋SlowPCC20那一项，不要弄混来源。

`same_record_target_I`明确声明本批模型实际使用目标同记录同步I；若其他模型使用不同输入，不能照抄这个来源。

## 4. 再重算原FM＋PCC20及其配对对照

实验组：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1
evaluation=$(cat "$run/pcc/latest_best_evaluation.txt")
out=/home/qht/huawei/results/B5-rescore-reliable-I/original_FM_pcc20_$(date +%Y%m%d_%H%M%S)

python -u scripts/rescore_reliable_i.py \
  --evaluation-dir "$evaluation" \
  --input-quality-source same_record_target_I \
  --output-dir "$out"

cat "$out/correlation_score.csv"
)
```

对照组使用同一命令，将`pcc/latest_best_evaluation.txt`改为`control/latest_best_evaluation.txt`，并将输出名original_FM_pcc20改为original_FM_slow_control20。

其余候选：使用各自已保存的evaluation目录，换--evaluation-dir和新输出名，不需要重训或重新采样。优先覆盖默认交叉组合seed42/43、原FM＋PCC/慢control、control100＋原S50-F、原B5-best；关键曲线重算各已保存epoch。

新范围应排除d12_f2efb4a4c95a40。当前数据版本：Task1唯一目标21不变，Task2机器39、体脂秤24，物理目标总数39。排除明细在每个task/input_qc_exclusions.csv。没有预测数组时，只有汇总CSV不足以准确重算。

## 5. 如果需要重新生成验证

更新后的B5-best validate默认新范围。旧模型包可直接使用，不必重新导出或改旧参考分数：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best
out=/home/qht/huawei/results/B5-best/validate_reliable_I_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best.validate \
  --config /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml \
  --device cuda \
  --output-dir "$out"
)
```

无新范围参考报告时，evaluation.json的reference_matched为null，且说明旧参考与当前范围不同；这不是模型出错。旧包SHA、组件来源和输入数据校验仍保留，不会静默跳过权重校验。

若要验证与第3节CPU重算一致，向上述命令额外传入同一模型新结果的`--reference-report /绝对路径/correlation_score.json --verify-reference`。

若仅复现旧0.698216等历史成绩，显式使用`--legacy-target-qc-only --verify-reference`；它不能当作新范围成绩。

## 6. 新训练与冻结实验

稳定B5/B5-best的新配置已选择main_correlation并设置独立输出目录。训练启动方式和初始化权重语法不变，仍需--execute-training；本次不要求马上训练。

新选模账本拒绝旧范围last.pt的精确resume，以免混用历史best。若以后需要继续训练，在新输出目录初始化兼容旧权重，或者另外实施可审计的优化器状态迁移；不要修改旧checkpoint协议字段强行恢复。

冻结的SlowPCC/V13等代码、study、cache不改。重评分只读取其结果数组。若后续要继续这些实验，应从更新后的稳定分支建立新study与独立运行，再采用新验证范围。

仅重算现有best/last不等于重选所有轮次。预测与权重齐全才能重选；缺失的历史权重不能恢复。先形成同口径候选表，再决定是否更新默认部署权重。
