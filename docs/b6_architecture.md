# B6 实现范围

分支 `B6`。详细设计、配置、服务器命令和验证记录见 [B6 README](../baselines/B6/README.md)。

本次在最新 B5 上新增独立 `baselines/B6` 与 `b6_*.yaml`，没有修改B5或main的数据预处理、评价实现，没有合并M1分支、导入M1回归训练器或载入M1权重。

参考骨干：`baseline/M1-main-v3` 的多尺度CNN、时间/导联双轴Transformer、按导联FPN。适配FM时新增11导联state编码、连续t注入、人口学FiLM与全分辨率局部state支路；输出速度场，再由与B5相同的ODE生成。

第一版只使用同步I与用户信息，慢走势附加权重0，历史设备条件后续单独实验。独立architecture ID和严格checkpoint校验，拒绝B5、M1、不同注意力轴或人口学配置直接续训。

骨干改变需要B6自己的公开预训练，不沿用B5公开checkpoint。10秒窗口合同保持不变；这不是跨窗口连续性修复，也不能由之前慢loss无收益就证明U-Net错误。
