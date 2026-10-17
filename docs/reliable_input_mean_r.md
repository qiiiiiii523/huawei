# 可靠输入I与两任务平均r

新运行的选模指标：`R=0.5×Task1 r+0.5×Task2 r`，保存`best_r.pt`，`last.pt`用于最新训练状态。Task2的r仍是机器与体脂秤各自评分后等权汇总。RMSE只作诊断，不计入R，也不用于平分时隐式决胜。

main只提供公共接口和协议，不实现各模型训练循环。其他分支需要显式接入；修改main配置不会自动改变旧模型的best保存逻辑。

## 输入与目标的质量分工

- 实际输入I明确不可靠：该配对的完整120秒、12个窗口一起排除。
- 输入I可靠、某个目标导联不可靠：只排除该目标导联的评分。
- 未标差但预测不佳：保留；不能按模型误差决定质量。
- 不可靠输入可以另作鲁棒性实验，不混入这套可靠输入验证。

```python
from ecg12gen.evaluate import evaluate_record_predictions, correlation_score_from_summaries

overall, details = evaluate_record_predictions(
    prediction_uV, target_uV, task_id, window_rows,
    target_quality_mask=target_mask,       # [N,12]，来自独立目标QC
    input_quality_mask=actual_input_mask,  # [N]或[N,1]，来自实际输入的QC
    require_reliable_input=True,
    input_quality_source="actual_anchor_QC_source",
)
# 两任务分别按完全相同的协议评分后：
score = correlation_score_from_summaries({"task1": task1_overall, "task2": task2_overall})
value_for_best_r = score["main_correlation"]
```

如果实际输入就是同一D12目标记录的I，可显式声明`input_quality_source="same_record_target_I"`，并传`target_mask[:,0]`。此声明会校验输入mask与目标I mask一致。共享函数不会自动作此假设；其他来源输入必须传自己的mask，不能拿目标I质量代替。

未开启`require_reliable_input`时保留旧行为；传了输入质量却未显式开启会报错，防止以为已经过滤。完整记录内mask不一致、某个必评分导联无有效目标、Task2某设备过滤后为空等情况会报错。

`evaluate_joint_anchor_predictions`启用新协议还需`apply_target_qc=True`。预测波形、原始μV、120秒拼接、重复配对在目标内平均、导联宏平均和Task2设备等权均不改变。

协议号：

- 实际输入质量：`explicit_reliable_anchor_record_v1`。
- 新记录评分范围：`record-r-reliable-input-target-qc-v1`。
- 新双任务选模：`mean-r-reliable-input-target-qc-v1`。

旧`competition_score`和`summarize_competition_score.py`仅保留作旧60分报告审计。新协议报告不输出RMSE加分；不要混用旧总分或仅目标QC的历史分数。

## CPU重算已保存预测

不加载模型，不训练，不采样；需要每个task目录内的prediction_uV.npy、target_uV.npy（或带SHA的target_reference.json）、target_quality_mask.npy、window_metadata.csv。

以原FM＋SlowPCC20的best评估为例，在更新后的main代码环境执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_eval_reliable_i

python scripts/rescore_reliable_i.py \
  --evaluation-dir /home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1/pcc/evaluation_best_20261016_193543_4025481 \
  --input-quality-source same_record_target_I \
  --output-dir /home/qht/huawei/results/B5-rescore-reliable-I/pcc20
```

`same_record_target_I`是调用者对原模型输入来源的明确声明，仅适用于实际使用该同步I的评估。其他输入来源示例：

```bash
python scripts/rescore_reliable_i.py \
  --evaluation-dir /absolute/old_evaluation \
  --input-quality-source independent_anchor_sidecar \
  --task1-input-quality-mask /absolute/task1_input_mask.npy \
  --task2-input-quality-mask /absolute/task2_input_mask.npy \
  --output-dir /absolute/new_report
```

输出目录必须为空，且不得位于原评估目录内部；日志如需tee，放在输出目录旁边。源数组前后SHA会核对。输出包括两任务诊断报告、排除清单、保留窗口身份、r1/r2/R以及rescore.json来源记录。重算结果不自动覆盖任何checkpoint，也不宣称已经重选所有轮次的best。

两份新报告可再次汇总：

```bash
python scripts/summarize_mean_r.py \
  --task1-overall /absolute/new_report/task1/overall_metrics.csv \
  --task2-overall /absolute/new_report/task2/overall_metrics.csv \
  --output-dir /absolute/new_summary
```

仅用于旧目标QC均值复核时，可显式加`--legacy-target-qc-only`；它会生成不同协议号，不能与可靠输入报告混合。

## 数据数量与历史候选

当前B5数据版本实际训练156条，I明确标差为0；验证40条物理目标中HNU11175/d12_f2efb4a4c95a40的I明确标差。过滤后预期Task1唯一目标21不变，Task2机器40→39、体脂秤25→24，两设备排除的是同一个物理目标，物理目标总数40→39。新报告的排除清单与统计用于核对，不能把设备数相加当作不同患者数。

准备继续比较的高分组合、seed42/43、SlowPCC/V13关键验证轮次需统一重算。公开数据fold9评分不因本次华为QC修改而重训。

若每轮预测齐全，可重算各轮重新选模；只有部分权重/预测，只能筛选现存候选。不能仅重命名best.pt或修改旧best账本。RMSE评分范围变化不要求改变训练损失。

## 同步与服务器操作

本地main提交推送到服务器使用的gitee后，首次创建独立评分工作树：

```bash
cd /home/qht/huawei
git fetch gitee main
git worktree add --detach /home/qht/huawei_eval_reliable_i FETCH_HEAD
```

已有该工作树时，在其中先检查未提交文件，再更新：

```bash
cd /home/qht/huawei_eval_reliable_i
git status --short
git fetch gitee main
git merge --ff-only FETCH_HEAD
```

main共享提交后，再在B5-best/B5显式接入。不要把整个B5-best并回main。当前提交不修改其他分支，因此旧训练与部署入口不会自动切换新评分范围。

SlowPCC等实验冻结了共享代码SHA，不要直接改旧study或cache校验来强行续跑。先使用独立main评分工作树处理保存数组；后续训练应建立新协议study、输出目录及best账本，明确初始化旧权重。更新默认部署时，保留旧参考分数，并建立带新协议的新参考报告。

## 验证

```bash
python -m unittest discover -s scripts -p 'check_*scoring.py' -v
python scripts/check_experiment_protocol.py
```

这些检查使用合成数据，不启动训练。实际服务器旧实验的重评分须另外执行，不能将合成检查当作新的模型成绩。
