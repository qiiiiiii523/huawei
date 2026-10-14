# 显式目标导联QC评分接口

main保留原默认评分。只有调用方显式传入`target_quality_mask`，才使用QC协议；B5在自己的分支启用此接口。旧模型默认调用不受影响。QC是本地验证协议，不代表主办方测试规则已经改变。

```python
overall, details = evaluate_record_predictions(
    prediction_uV, target_uV, task_id, window_rows,
    target_quality_mask=quality_mask,  # [N,12]，1/True为正常
)
```

mask按I、II、III、aVR、aVL、aVF、V1—V6排列，来自独立的设备质量标记，不能根据模型预测好坏生成。同一物理目标的12个窗口、重复配对和两设备必须使用一致的mask；不一致会报错。

- 先拼接每条配对的真实120秒。对每个导联，排除该导联被标差的目标，再在目标内平均配对指标、跨目标等权汇总；II—V6导联等权得到r。
- RMSE只累积该导联正常目标的误差，并使用这些目标的样本数。重复配对先在目标内平均平方误差，不重复增加目标权重。
- Task2两设备分别计算r和V1—V6平均RMSE，再等权汇总。加分先按每设备RMSE换算，再平均。
- 总分仍为`50 × (0.5r1 + 0.5r2) + 两设备平均加分`，最高60分。
- 不删整条记录，不把坏导联置零，不做去均值或目标校准。输入仍须满足原始有限μV数组的契约。
- 若某个必评分导联完全没有正常目标，或其所有正常目标r均未定义，直接报错，不虚构分数或悄悄减少评分导联。

新协议：`correlation50_bonus10_target_record_qc_v3`。每导联报告增加r、正常目标/样本数、QC排除目标/配对数。总报告中的目标总数包含保留的整条记录，不等于每个导联的有效目标数。

旧报告不能与新QC分数直接混比。`scripts/summarize_competition_score.py`支持同协议报告，拒绝混合完整导联报告和QC报告。

对已有数组显式重算（不生成波形、不训练）：

```bash
python -m ecg12gen.evaluate \
  --prediction prediction_uV.npy \
  --target target_uV.npy \
  --metadata window_metadata.csv \
  --target-quality-mask target_quality_mask.npy \
  --task-id task2 \
  --output-dir new_qc_report
```

`evaluate_joint_anchor_predictions`即便原来传了训练mask，也仍保持默认完整导联评分。启用QC需额外设置`apply_target_qc=True`，并提供mask。
