# 体脂秤患者内 I→II 导联关系实验：本地实现与远程运行计划

## 研究问题

旧的整记录注意力、心拍、节律编码器未在正确配对相对跨患者打乱的 `r_missing11` 上显示稳定收益。新变体单独检验：体脂秤一次记录中**同步采集**的 I、II 导联是否能教模型形成患者特异的 I→II 心拍映射，并让该关系作为条件帮助 B5 利用**目标时刻真实 I 导联**预测其余导联。

体脂秤与目标 d12 仍为不同时间、不同设备记录。本实验不逐拍跨设备对齐，不把体脂秤 II 当成目标时刻 II，也不在推理时读取目标 II。

## 改动与防泄漏设计

- `baselines/B5/context_body_lead_relation.py` 新增 `lead_relation` 编码器。仅在体脂秤 I 上检测 QRS，截取同一体脂秤记录中同步的 I、II 心拍，最多 32 个；**只用 I 的 RMS** 对每拍 I、II 同尺度归一化，保留相对幅值且不让留出 II 的幅值泄漏进输入 I。
- 偶数序号心拍的 I、II 作为支持集，生成 128 维患者关系向量；奇数序号心拍的 I 为查询，由该向量生成患者条件的短时滤波核，预测对应的体脂秤 II。查询 II 只出现在辅助损失中，不进入支持集编码或预测器输入。至少需要 2 个支持心拍和 2 个查询心拍。
- 关系向量经小门控加入旧注意力 context embedding，仍通过原 FiLM 与 B5 的目标 I 条件联合工作。这是**关系条件化**的第一步，不是把生成的体脂秤 II 波形直接作为目标 II；能否迁移到目标时刻需用验证集检验。
- 训练从 `A_frozen_attention_v1/best.pt` 的 EMA 初始化；原 B5、注意力编码器和 FiLM 全部冻结，只训练关系编码器、I→II 映射头和门控。原先的 beat/RR 分支不参与本实验。总目标为原 flow loss 加 `0.02 ×` 留出心拍 I→II 平滑 L1 损失；权重是预先固定的试验设置，不根据验证结果临时调节。
- 独立输出 `results/context_body/A_frozen_lead_relation_v1/`，不覆盖已有 `attention`、`beat`、`rhythm` 或 `beat_rhythm` 结果。仍用 `A_raw_window`、seed 42、10 epoch、batch 4、累积 2、第 5/10 epoch 的 EMA/Heun16/K16 验证，以 `r_missing11` 选择最佳权重。

## 已完成的本地检查

`python scripts/check_b5_body_lead_relation.py` 的 CPU 小尺寸真实数据预检通过：可抽取有效心拍；留出 II 只改变监督损失；关系编码器、映射头和门控梯度非零。该检查不等于完整训练或性能验证。本地 B5 基础权重是占位文件，正式的 checkpoint 一致性及完整 flow 前向／梯度预检仍需在远程 GPU 环境运行。

## 远程连接后执行

在包含既有 B5 基础权重、注意力最佳权重、预处理参数和体脂秤缓存的远程项目根目录，设置 `PYTHONPATH=.` 后：

```bash
python scripts/train_b5_body_frozen_adapter.py --encoder-variant lead_relation --device cuda --preflight
python scripts/train_b5_body_frozen_adapter.py --encoder-variant lead_relation --device cuda
python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_lead_relation --mode A_true --device cuda
python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_lead_relation --mode A_disabled --device cuda
python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_lead_relation --mode A_shuffled --device cuda
```

先确认 preflight 打印 `relation_valid_records > 0`、辅助损失有限且关系编码器梯度非零，FiLM 梯度为零；若不满足，不启动正式训练。训练后重点比较：同一模型正确、禁用、跨患者打乱时的 `r_missing11`，同时查看逐导联指标中 **II** 及其余肢体导联是否有变化；V1–V6 单独报告，不预设胸导联必然受益。若正确配对未优于打乱，不能因为辅助 I→II 学习成功就宣称任务改善。

权重、数据、预测数组不应上传 GitHub。训练完成前不要关闭 GPU 实例；训练／评估全部退出后再提醒用户自行关卡。
