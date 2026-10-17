# 体脂秤 I→II 导联关系实验结果（2026-10-17）

## 设置

- 从固定 B5 与 `A_frozen_attention_v1/best.pt` 初始化；冻结 B5、原注意力编码器和 FiLM，仅训练新增关系编码器、I→II 映射头及门控。
- 体脂秤同步 I/II 心拍按偶/奇序号分为支持与查询；用查询 I 预测查询 II 的辅助损失权重为 0.02。目标时刻仅使用真实 I，不用目标 II 或跨设备逐拍对齐。
- `A_raw_window`，seed 42，10 epoch，batch 4、梯度累积 2；第 5/10 轮以 EMA、Heun16/K16 验证，按 `r_missing11` 选 best。
- 远程 SSH `01`，项目 `/root/autodl-tmp/Huawei_B5_context/huawei`；定向上传实验文件，**没有执行 git pull**。训练与诊断已完成，进程退出。

## 训练与诊断

GPU 预检通过：有效关系记录 4 条，关系模块梯度非零，FiLM 梯度为零。第 5 轮得到最佳 `r_missing11 = 0.6120384903`；第 10 轮未超过。`best.pt` 为第 5 轮模型。

| 同一最佳模型的输入条件 | `r_missing11` ↑ | V1–V6 平均 RMSE (µV) ↓ |
|---|---:|---:|
| 正确配对 `A_true` | 0.6120384903 | 616.0005 |
| 禁用 context `A_disabled` | 0.6126322535 | 623.3514 |
| 跨患者打乱 `A_shuffled` | 0.6120330947 | 615.8225 |

正确配对 − 打乱的 `r_missing11` 仅为 **+0.000005396**；正确配对 − 禁用为 **−0.000593763**。正确配对相对禁用的胸导联平均 RMSE 较低约 7.35 µV，但相对打乱反而较高约 0.18 µV。这不是可靠的患者特异迁移收益；不能把辅助 I→II 学习成功等同于目标 11 导联预测获益。

## 后续判断

这版关系向量通过旧 FiLM 间接调节 B5，没有把学到的 I→II 映射显式应用到目标时刻 I 波形。若继续研究，应先检查关系嵌入的患者区分性、真实/打乱条件下 FiLM 差异及导联 II 的逐导联指标，再决定是否尝试显式目标 I→II 候选或新的接入方式。不要仅凭这次辅助损失下降而加大门控。

远程日志：`results/context_body/A_frozen_lead_relation_v1_train.log` 与 `results/context_body/A_frozen_lead_relation_v1_diagnostics.log`。远程权重：`results/context_body/A_frozen_lead_relation_v1/best.pt`。本地只保留本结果文档，不下载权重或预测数组。
