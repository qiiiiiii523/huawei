# ECG-12 joint-anchor 公共主干

`main` 是供组员创建 baseline 分支的公共底座。它提供数据、预处理、loss、评估和检查；**不提供网络、训练循环、checkpoint 或预测结果**。

## 1. 固定任务：模型到底要做什么

```text
task1: watch context + machine-I anchor -> d12 target
task2: d6 context    + machine-I anchor -> d12 target
```

- **anchor**：目标时刻机器 I 导联，与 d12 target 同记录、同窗口、严格同步。
- **context**：同一受试者的跨时刻辅助 ECG。task1 是 watch；task2 是 machine d6 或 body-scale d6。
- **target**：目标时刻的 d12 ECG。

只有 anchor 与 target 同步。context 不是 target 时刻同步波形，不能按采样点与 target 比较或拼成同步多通道 ECG。

## 2. main 已提供什么

| 内容 | 位置 | 组员如何使用 |
|---|---|---|
| 严格预训练数据 | `ecg12gen/d12_pretrain.py` | `d12 I -> d12`，只读 train-only strict index |
| joint 数据 | `ecg12gen/dataset.py` | 读取 `context_ecg`、`anchor_i_ecg`、`Y_12lead` 和独立 masks |
| task2 A/B | `ecg12gen/body_scale.py` | body-scale A/B、五导联 context 消融 |
| 推理输入检查 | `prepare_joint_anchor_inference()` | 测试必须显式传入 machine-I anchor；没有 target 参数 |
| 预处理 | `ecg12gen/preprocessing.py` | train-only frozen scale、per-window median baseline |
| loss | `ecg12gen/losses.py` | `joint_anchor_sync_loss()`、`replace_output_i_with_anchor()` |
| V0 评估 | `ecg12gen/evaluate.py` | raw-uV official V0、task2 分设备/subject/V1–V6 诊断 |
| 配置与检查 | `configs/`、`scripts/` | 统一实验契约与提交前 smoke check |

## 3. baseline 分支必须实现的网络模块

每个 B0/B1/B2/M1 分支都应实现下列模块；模块内部结构可以不同，但输入语义不能改变。

| 模块 | 必须做什么 | 不允许做什么 |
|---|---|---|
| **Anchor encoder + d12 decoder** | 输入 machine-I anchor，输出完整 d12；先完成 strict `d12 I -> d12` 预训练 | 用 validation d12 预训练 |
| **Context encoder** | 编码 watch/d6 的跨时刻个体、设备、形态信息 | 将 context 当作 target-time 同步导联 |
| **Fusion module** | 融合 anchor representation 与 context representation；可用 concat、gate、cross-attention 等 | 按相同采样点把 context 与 anchor 拼成同步 ECG |
| **d12 output head** | 训练时输出完整 `[batch,12,5000]`，包括模型预测的 I | 从真实 target 读取 I 或其他导联 |
| **可选 baseline head** | 预测 raw-uV 合成所需 d12 baseline | 推理时读取真实 target baseline |

建议统一模型接口：

```python
d12_prediction = model(
    context_ecg,          # task1 [B,1,5000]；task2 [B,6/5,5000]
    anchor_i_ecg,         # [B,1,5000]
    context_lead_mask,
    anchor_lead_mask,     # target-time 仅 I=true
)
```

## 4. 必须保留的 baseline 对比

核心问题是：跨时刻 context 是否在同步 I anchor 之外带来增益。因此所有模型分支至少保留以下两个可比较实验：

| 实验 | 输入 | 目的 | 是否正式合法 |
|---|---|---|---|
| **Anchor-only** | `machine-I anchor -> d12` | 严格预训练主干 / 无 context 基线 | 是 |
| **Joint-anchor** | `context + machine-I anchor -> d12` | 检验 context 融合带来的增益 | 是 |

两者必须使用相同 subject split、strict 初始化、预处理、训练预算、loss 和 raw-V0 checkpoint 规则。报告中必须同时给出 anchor-only 与 joint-anchor 的 task1 r1 / task2 r2、RMSE、task2 V1–V6 和 machine/body 分层结果。

可选但推荐的 task2 消融：

- context 全 d6 `[I, II, III, aVR, aVL, aVF]` 对比五导联 `[II, III, aVR, aVL, aVF]`；
- body-scale A 对比 B；
- 不同 fusion 模块对比。

**不要做 context-only 的逐点 d12 重建训练**：没有同步 anchor 时，context-target 逐点监督不合法。

## 5. 训练流程

### 阶段 1：严格 anchor 预训练

```text
输入：train-only d12 的 I
标签：同窗口 d12
目标：训练 Anchor encoder + d12 decoder
```

```python
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.contracts import SupervisionMode

strict_train = StrictD12PretrainDataset(
    "configs/common.yaml", SupervisionMode.D12_I_PRETRAIN.value
)
```

严格 Dataset 只使用 `metadata/d12_strict_pretrain_index.csv`，不得读取 validation d12。

### 阶段 2：joint-anchor adaptation

```text
输入：cross-time context + target-time I anchor
标签：同一 target-time d12
目标：保留 anchor 主干，同时训练 Context encoder + Fusion module
```

```python
from ecg12gen.dataset import JointAnchorDataset

train = JointAnchorDataset("configs/common.yaml", "task1", "train")
sample = train[0]
# sample.context_ecg
# sample.anchor_i_ecg
# sample.Y_12lead
# sample.context_lead_mask
# sample.anchor_lead_mask  # 仅 I=true
```

train/validation 的 `anchor_i_ecg` 从该窗口 target I 模拟构造，metadata 会标记：

```text
simulated_from_target_i_for_test_available_input
```

这仅模拟测试可见的输入；模型推理函数不得接受 `Y_12lead` 或 target NPY。

### loss

```python
loss = joint_anchor_sync_loss(prediction, target, anchor_i)
prediction = replace_output_i_with_anchor(prediction, anchor_i)  # 可选
```

loss 包含：

- 完整 12 导联的 Huber + PCC；训练不覆盖模型预测 I；
- 完整 d12 的导联代数约束；
- I 的 observed consistency。

逐点损失的合法性来自 `anchor_i_ecg <-> Y_12lead` 严格同步，**不是**来自 context。

## 6. 预处理：模型分支必须遵守

1. 使用 `ECGPreprocessor`；不要自己重新定义归一化。
2. 固定 μV、500 Hz、10 秒、5000 点和 canonical d12 顺序。
3. 每窗每导联减 median baseline。
4. scale 只在 train 拟合，validation/test 必须复用冻结实例。
5. watch、machine d6、body-scale d6、machine-I、d12 使用各自 source scale；machine-I scale 只能来自 train d12 的 I。
6. raw-uV 其他导联只能用模型预测 baseline 合成，绝不能读取真实 target baseline。

```python
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig

preprocessor = ECGPreprocessor.fit(
    PreprocessingConfig.from_yaml("configs/preprocessing.yaml"),
    train_signals,
)
```

## 7. validation、测试与评估

validation 必须模拟测试接口：

```text
task1 validation: watch context + validation d12 I -> validation d12
task2 validation: d6 context    + validation d12 I -> validation d12
test:             organizer context + organizer machine-I -> prediction
```

validation 的 target 只用于：构造模拟可见 anchor、计算 loss、离线 V0 评分；不能作为模型输入的 hidden target。

checkpoint 按 validation **official raw-uV V0** 选择。centered diagnostic 只用于定位形态/基线问题，不能替代 official V0。

每次 validation 必须保留三套 r：

| 指标 | 预测 | 用途 |
|---|---|---|
| `r_raw_12` | 模型原始完整 d12 输出 | 检查完整预测和 I 身份保持 |
| `r_submit_12` | 仅在验证/模拟提交时用输入 anchor 覆盖预测 I | 最接近正式测试的官方成绩；用于 checkpoint 选择 |
| `r_missing11` | 原始预测的 II、III、aVR、aVL、aVF、V1–V6 | 衡量真正缺失 11 导联的重建能力 |

`pred_submit` 的 I 覆盖不是训练策略：训练时必须保留模型完整 d12 输出和 I 的监督；只有 validation/test 输出阶段才执行 `pred_submit[:, 0:1] = anchor_i_ecg`。

```powershell
python -m ecg12gen.evaluate `
  --prediction results/task1_validation_prediction.npy `
  --anchor results/task1_validation_anchor_i.npy `
  --target ../task1_output/task1_validation_target.npy `
  --metadata ../task1_output/task1_window_metadata.csv `
  --task-id task1 `
  --output-dir results/task1 `
  --write-centered-diagnostic
```

task2 必须额外保留 machine/body 分层、subject-macro 和 V1–V6 RMSE。评估报告应写入 `evaluation_input_contract=joint_anchor_test_like`。

## 8. 从 main 开始工作的最小清单

```text
1. 从 main 创建 baseline 分支。
2. 运行共享检查。
3. 实现 Anchor encoder + d12 decoder，并跑严格预训练。
4. 先报告 Anchor-only baseline。
5. 添加 Context encoder + Fusion module，跑 Joint-anchor adaptation。
6. 在相同预算下比较 Anchor-only vs Joint-anchor。
7. 仅以 raw-V0 validation 选择 checkpoint；记录 task2 分层诊断。
8. 测试时只传 context + 主办方 machine-I anchor。
```

共享检查：

```powershell
python scripts/check_d0_d1_v0.py
python scripts/check_experiment_protocol.py
python scripts/check_preprocessing_protocol.py
python scripts/check_body_scale_variants.py
```

正式配置：

- `configs/experiments/task1_joint_anchor.yaml`
- `configs/experiments/task2_joint_anchor.yaml`
- `configs/training_protocol_v1.yaml`
- `configs/losses.yaml`

## 9. 统一 context 融合实验标准

`configs/context_fusion_protocol.yaml` 是 B1、B2、B3、M1 共用的实验命名和公平比较标准；main 只定义协议，不实现任何 FiLM、gate 或 residual 网络模块。

统一实验阶段为：

| 阶段 | 输入/输出 | fusion_mode | 初始化 |
|---|---|---|---|
| `P0_anchor_only` | machine I(C) → machine d12(C) | `none` | train-only strict pretraining，从头开始 |
| `P1-C1` | context + machine I(C) → machine d12(C) | `film` | 同一网络架构且结构配置兼容的 P0 checkpoint |
| `P1-C2` | context + machine I(C) → machine d12(C) | `gated_residual` | 同上 |
| `P1-C3` | context + machine I(C) → machine d12(C) | `film_gated_residual` | 同上 |

C1、C2、C3 的定义、gate/residual 初始化常量和 checkpoint 兼容字段固定见该 YAML。所有 P1 使用 `joint_anchor_sync_loss`、固定 subject split、frozen preprocessing、同任务同架构同训练预算和 validation raw-uV V0 checkpoint 规则。训练始终输出完整 d12，不回填 I；只有 validation/test submit 输出才回填真实 anchor I。

Task 1 固定为 `watch I(A) + machine I(C) -> machine d12(C)`。Task 2 分别运行互斥的 machine/holter d6(B) 与 body-scale d6(A) source variant；不实现、不声明、不比较 `P1-both`。context 仅为条件信息，禁止 context-target 逐点损失、跨时刻波形硬对齐、R 峰伪配对和训练阶段 I 回填。

B0 不要求 C1/C2/C3，可保留 P0 或单独记录 linear-context diagnostic。B1/B2/B3/M1 至少比较 P0 与 P1-C3；B2 和 M1 完成 C1/C2/C3 模块消融。实验记录必须填写 architecture_id、architecture_config_hash、P0 checkpoint、task/context source、fusion_mode、训练预算、r_raw_12、r_submit_12、r_missing11、RMSE 及 shuffled-context 结果；Task 2 还需记录 machine/body 分层、subject-macro 和 V1–V6 诊断。

不要向 main 提交原始 ECG、窗口 NPY、checkpoint、预测、患者级结果或训练日志。实验详情记录到 `docs/experiment-record-template.md`。
