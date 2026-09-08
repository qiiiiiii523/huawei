# Joint-anchor 实验记录模板

## 基本信息

- 实验编号、日期、执行人、Git commit：
- 任务：task1 / task2；模型名称/版本：
- protocol stage：`P0_anchor_only` / `P1-C1` / `P1-C2` / `P1-C3`；fusion_mode：
- architecture_id：
- architecture_config_hash：
- 唯一变化：

## 固定数据契约

- strict stage：`d12 I -> d12`，train-only strict index：
- context source/type：
- Task 2 context source variant：`ecg_machine_d6` / `body_scale_d6`（互斥；不得记录 P1-both）
- context channel variant：
- anchor source：`ecg_machine_i`
- context-target relation：`same_subject_cross_time`
- anchor-target relation：`same_record_same_window`
- anchor_available_at_test：true
- subject split、seed、质量门控、预处理 frozen scale：

## 训练与验证

- P0 strict checkpoint / P1 是否加载该权重：
- P0 checkpoint path/id：
- P0 checkpoint architecture_id / architecture_config_hash compatibility：
- joint stage loss：完整 d12；context-target 逐点损失：forbidden；训练 I 覆盖：forbidden
- fusion definition：C1 FiLM / C2 gated residual / C3 FiLM then gated residual
- gate/residual initialization：
- training budget：epochs / batch size / optimizer / learning rates / seed：
- body-scale variant（task2）：
- checkpoint：validation test-like joint-anchor official raw-V0
- test-like validation result：

## 结果和诊断

- r_raw_12 / r_submit_12 / r_missing11：
- RMSE：
- task1 r1 / task2 r2；12 导联 r、RMSE：
- task2 machine/body、subject-macro、V1–V6 RMSE：
- shuffled-context result（r_submit_12 / r_missing11 / RMSE）：
- raw-V0 与 centered diagnostic：
- 异常、失败和可比性备注：
