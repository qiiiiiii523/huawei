# Joint-anchor 实验记录模板

## 基本信息

- 实验编号、日期、执行人、Git commit：
- 任务：task1 / task2；模型名称/版本：
- 唯一变化：

## 固定数据契约

- strict stage：`d12 I -> d12`，train-only strict index：
- context source/type：
- context channel variant：
- anchor source：`ecg_machine_i`
- context-target relation：`same_subject_cross_time`
- anchor-target relation：`same_record_same_window`
- anchor_available_at_test：true
- subject split、seed、质量门控、预处理 frozen scale：

## 训练与验证

- P0 strict checkpoint / P1 是否加载该权重：
- joint stage loss：完整 d12；context-target 逐点损失：forbidden；训练 I 覆盖：forbidden
- body-scale variant（task2）：
- checkpoint：validation test-like joint-anchor official raw-V0
- test-like validation result：

## 结果和诊断

- r_raw_12 / r_submit_12 / r_missing11：
- task1 r1 / task2 r2；12 导联 r、RMSE：
- task2 machine/body、subject-macro、V1–V6 RMSE：
- raw-V0 与 centered diagnostic：
- 异常、失败和可比性备注：
