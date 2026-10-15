# 体脂秤 context 编码器第二轮实验准备

本轮仅改 `body_encoder`，从原始固定 B5 EMA 重新初始化；B5 主干继续冻结，原 FiLM 结构与零初始化保持不变。旧编码器、旧实验权重和结果不修改。

新编码器沿用原逐窗口 masked CNN 和投影层，在相同 seed 42 下复制它们与前一轮一致的初始参数。对每条历史体脂秤 ECG 的窗口特征，按有效采样点权重计算逐维加权均值和标准差，再用小型融合层产生 embedding。跨设备波形仍只作为不对齐的 context，不直接充当目标导联；窗口顺序对汇总结果无影响。

独立实验名为 `A_frozen_meanstd_v1`。训练协议与 `A_frozen_adapter_v1` 一致：体脂秤 `A_raw_window`、10 epoch、batch 4、累积 2、学习率 1e-4、context dropout 0.1、seed 42；第 5/10 轮按 25 配对/300 窗口的 `r_missing11` 选最佳，Heun16/K16。之后用同一最佳权重做禁用 context、跨患者打乱 seed 101/202，以及 embedding/FiLM 审计。**不以 embedding 更分散本身作为成功标准**；主要看真实配对是否稳定优于禁用和打乱，同时记录胸导联 RMSE。

本地合成数据检查、真实数据 CPU 单批次预检和远程 GPU 单批次预检均已通过。代码已上传到 04，10 轮训练及禁用/打乱 context 对照、适配器审计均已完成；结果与结论见 `docs/context_body_meanstd_results.md`。
