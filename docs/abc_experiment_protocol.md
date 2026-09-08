# Joint-anchor 实验协议

本协议定义 main 的共享流程，不定义任何具体网络、训练循环或 checkpoint。

严格阶段是 `d12 I -> d12`，仅使用 train-only、按 `target_record_id + window range` 去重的 strict index。它初始化测试时的 anchor 主干。

适配阶段是 `context ECG + machine I anchor -> d12 target`。context 是跨时刻的个体、设备和形态条件；task1 为 watch ECG，task2 为 machine/body-scale d6。anchor 是 target 时刻的时序锚点，和 target 严格同记录、同窗口同步。

严格预训练和 joint-anchor 微调都输出、并对完整 d12 计算训练损失；训练时不得将输出 I 硬替换为 anchor。context-target 逐点 Huber/MSE/PCC、raw weak 频谱统计、pair-invariant loss、R 峰伪配对和时间 warp 都不属于该协议。逐点训练监督的合法性来自 anchor-target 严格同步。

P0 是 anchor-only 严格主干；P1 是 context-conditioned，必须加载 P0 权重。train、validation、test 输入同构。前两者以 target I 构造可见输入模拟 anchor；test 由主办方显式传入 machine-I anchor。validation 保留 r_raw_12、r_submit_12、r_missing11；仅 r_submit_12 在输出阶段覆盖 I，作为最接近正式测试的官方成绩。task2 报告需保留 machine/body 分层、subject-macro、V1–V6 RMSE 与 raw-V0 / centered diagnostic 的区别。
