# main 分支 A/B/C 实验协议（修订版）

本协议只定义共享基础设施；不提交 checkpoint、训练结果、原始 ECG 或大体积派生 NPY。

## 共同边界

- A/B/C 是训练策略；B0/B1/B2/M1 是模型架构。兼容模型必须读取相同实验 YAML，不复制公共数据/评价逻辑。
- 所有实验引用 `common.yaml`、`preprocessing.yaml`、`training_protocol_v1.yaml`、`losses.yaml`，不复制采样率、导联顺序等全局真值。
- B0/B1/B2 使用无 adapter 的统一预处理；M1 先跑 no-adapter，只有 task2 分设备 Centered 诊断变差才验证 adapter。
- 官方选择指标始终是 raw-μV V0：r1、r2、任务二 V1–V6 RMSE；Centered 指标仅定位形态、baseline 与设备域问题。

## 严格 d12 预训练

`metadata/d12_strict_pretrain_index.csv` 由 `scripts/build_d12_strict_pretrain_index.py` 生成。它只保留 train subject、usable 窗口，并以 `target_record_id + start/end_sample_500hz` 去重。d12-I、d12-six 预训练只能读取该索引，绝不读取 validation d12。

## 实验矩阵

| 臂 | task1 | task2 |
|---|---|---|
| A | 原始 watch→d12 弱配对适配 | machine/body d6→d12 弱配对适配 |
| B | 严格 d12-I 预训练 + 弱配对混合适配 | 严格 d12-six 预训练 + 弱配对混合适配 |
| C | 严格 d12-I 预训练 + R 峰伪配对混合适配 | 无 C 路线 |

配置位于 `configs/experiments/`。A 的弱配对 loss 不得逐条比较其配对 d12；它使用可见导联一致性、完整导联生理约束和**独立 strict-train d12 reference bank**的批级频谱统计，避免把不同时刻的 d12 当作逐点标签。任何模型实现 A 前必须记录其防塌缩目标及 reference bank 采样规则。

## 损失合同

- 严格同步：missing leads 上的 Huber/PCC；完整 12 导联生理约束；低权重 observed consistency。
- 原始弱配对：禁止与同一行配对 d12 计算逐点 Huber/MSE/PCC；只允许 observed consistency、独立 d12 reference bank 的统计约束与生理约束。
- R 峰伪配对：`physical_sync=false`；只有 `accepted=true`、带 `alignment_quality_score` 的样本才允许 `pseudo_pointwise_loss_allowed=true`，且按质量加权。

`calibrate_loss_weights.py` 仅汇总 train dry-run 的最多 200 个 batch，不自动改写权重；权重确认后才人工更新 `losses.yaml`。

## task1 R 峰伪配对 C1

只读取 task1 train 窗口，且绝不覆盖 `task1_output`。检测信号为去 median baseline 的 watch 与 d12-I；检测器为 `wfdb_xqrs`。心搏以 `[R-200,R+300)` 截取 500 点，**截取阶段不重采样**。

候选心搏采用顺序保持、一对一动态规划匹配，代价包含形态相关、RR 差异和 QRS 宽度差。通过质量门控后，以匹配 R 峰构造 watch-time → d12-time 的单调映射。生成 pseudo d12 时，才以线性插值将原始 d12 全部 12 导联映射到 watch 的 500 Hz、5000 点栅格；全导联共享同一映射。若映射越界或不单调则拒绝窗口。

派生产物仅位于被忽略的 `task1_rpeak_pseudo_output/`：输入仍为 `[N,1,5000]`，伪 target 为 `[N,12,5000]`。manifest 记录每个候选心搏对及 `accepted`、`alignment_quality_score`、`loss_weight`、检测器版本；只有 accepted 窗口可进入 C1。

## 架构接入

- B0：严格同步 Ridge 映射后迁移到 task1/task2 validation，必须输出 V0 结果；不实现复杂弱配对/C1 训练。
- B1/B2/M1：可读取 A/B/C1 配置；B2 的随机导联 mask 仅用于 strict d12 同步增强。
- 所有架构必须保留 `lead_mask`、`missing_mask`、`supervision_mode`、`alignment_mode`、`alignment_quality_score`、`pointwise_loss_allowed`（伪配对为 `pseudo_pointwise_loss_allowed`）。

## 必须检查

形状、互补 mask、subject split、validation d12 未进入 strict/R 峰构造、严格索引去重、R 峰一对一/单调/不越界/12 导联同映射、仅 accepted 进入 C1，以及 V0 task1/task2 和 task2 分设备 subject-macro 报告。
