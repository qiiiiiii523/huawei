# Joint-anchor 实验协议

本协议定义 main 的共享流程，不定义任何具体网络、训练循环或 checkpoint。

严格阶段是 `d12 I -> d12`，仅使用 train-only、按 `target_record_id + window range` 去重的 strict index。它初始化测试时的 anchor 主干。

适配阶段是 `context ECG + machine I anchor -> d12 target`。context 是跨时刻的个体、设备和形态条件；task1 为 watch ECG，task2 为 machine/body-scale d6。anchor 是 target 时刻的时序锚点，和 target 严格同记录、同窗口同步。

严格预训练和 joint-anchor 微调都输出、并对完整 d12 计算训练损失；训练时不得将输出 I 硬替换为 anchor。context-target 逐点 Huber/MSE/PCC、raw weak 频谱统计、pair-invariant loss、R 峰伪配对和时间 warp 都不属于该协议。逐点训练监督的合法性来自 anchor-target 严格同步。

P0 is the strict anchor-only backbone; P1 is context-conditioned and must load compatible P0 weights. Train, validation, and test use the same input contract: train/validation simulate the visible anchor from target I, while test receives organizer-provided machine I.

Validation retains `r_raw_12`, `r_submit_12`, and `r_missing11`. Official `task1_r1` / `task2_r2` and checkpoint selection use raw-uV `r_missing11` over II--V6. Anchor-I replacement is diagnostic only and is not scored.
