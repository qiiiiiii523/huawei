# 按目标记录等权评分

评分按 `target_record_id` 汇总，不按受试者ID丢弃真实的不同采集，也不删除配对或修改预测。

1. 每个pair仍拼接12个10秒窗口，得到真实120秒记录，按原始μV计算11个缺失导联的r。
2. 同一target有多个pair时，每导联先平均这些pair的r，再对独立target等权平均，最后平均11导联。不是先平均预测再算r，也不挑选最佳pair。
3. RMSE先在同一target内平均各pair的平方误差，再按目标点数汇总、开方并平均导联；不能直接平均各pair的RMSE。Task2加分仍只用V1–V6。
4. Task2在机器、体脂秤两组内部独立做上述汇总，再固定1:1合并r和各设备分别换算的加分。不跨设备删掉同一target。
5. 同一target的原始波形或subject_id不一致时报错，避免把错误缓存静默合并。undefined r先在target内排除无定义pair，整条target该导联均无定义时才在目标平均中排除，并报告计数。

网络、训练loss、输入条件、训练去重、scale、窗口长度、K和推理接口均不因此改变。公式仍为50分相关性主分＋10分加分。协议标签更新为 `correlation50_bonus10_target_record_v2`，设备汇总协议为 `machine_body_equal_weight_target_record_score_v2`；旧配对权重报告不能混入新总分。

`n_records`为兼容既有报告保留配对记录数量，新增`n_pairs`和`n_unique_target_records`明确说明评分权重。Task2汇总的`n_unique_target_records`是两设备各自目标数量之和；`n_unique_physical_target_records`为两组目标ID的并集大小。RMSE的`n_validation_points`按目标统计有效评分点数，不重复计算配对点数。

运行合成数据检查（不训练）：

```bash
python scripts/check_target_record_scoring.py
```

所有调用共用`evaluate_record_predictions`的模型和B5周期验证自动使用新口径，不需重建缓存。已有模型可在相同验证集重新评估；评分变化不使旧权重失效。新旧r和best阈值不直接比较，B5禁止把旧评分状态原样resume进新协议。需要后续训练时，以兼容权重初始化新运行并重新建立best状态。

这是项目统一采用的验证汇总协议；主办方最终计分仍以其评分程序为准。
