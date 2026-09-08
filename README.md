# ECG-12 joint-anchor 公共主干

## B2 v1 历史说明（已被下方 B2 v2 替代）

本分支已在 `main` 公共契约之上实现 B2 的两阶段 baseline；它不再支持旧的 weak、A/B/C 或 R 峰伪配对路线。

| B2 模块 | 文件 | 作用 |
|---|---|---|
| Anchor encoder + d12 decoder | `ecg12gen/b2_model.py` | P0 只从 machine-I anchor 预测完整 d12 |
| Context encoder | `ecg12gen/b2_model.py` | 分别编码 watch / d6 跨时刻条件 |
| Gated fusion | `ecg12gen/b2_model.py` | `anchor_tokens + gate × context_tokens`；不把 context 当同步通道 |
| 数据适配 | `ecg12gen/b2_data.py` | 只读 strict index 与 `JointAnchorDataset`，并调用共享 frozen preprocessor |
| P0/P1 训练 | `ecg12gen/b2_train.py`、`scripts/train_b2.py` | P1 强制加载 P0 权重；训练完整 d12，不覆盖模型预测 I |
| 预测 / 验证 | `scripts/predict_b2.py` | 外部推理必须传入显式 machine-I anchor；validation 使用 test-like 模拟输入 |

### B2 的运行顺序

先训练 P0。它只读取 `metadata/d12_strict_pretrain_index.csv` 的 train-only d12 I，并在 task 的 validation joint-anchor 数据上报告 anchor-only 下限：

```powershell
python scripts/train_b2.py --experiment T1-P0 --task-id task1 --output-dir results/b2_task1_p0
python scripts/train_b2.py --experiment T2-P0 --task-id task2 --output-dir results/b2_task2_p0
```

再训练 P1。`--p0-checkpoint` 必填；P1 才会启用跨时刻 context encoder 和 fusion。task2 可选 body-scale B 或五导联 context 消融：

```powershell
python scripts/train_b2.py --experiment T1-C3-watch --task-id task1 --p0-checkpoint results/b2_task1_p0/b2_best.pt --output-dir results/b2_task1_p1
python scripts/train_b2.py --experiment T2-C3-machine --task-id task2 --p0-checkpoint results/b2_task2_p0/b2_best.pt --context-channel-indices 1 2 3 4 5 --output-dir results/b2_task2_p1_d6_without_i
```

validation 用 public Dataset 从 validation target 的 I 模拟开放的测试 anchor；模型不会收到 target。该命令会保存 `prediction_raw.npy`、`prediction_submit.npy` 和三套 r 的 raw-uV V0 报告：

```powershell
python scripts/predict_b2.py --checkpoint results/b2_task1_p1/b2_best.pt --task-id task1 --validation --output-dir results/b2_task1_p1_validation --centered-diagnostic
```

正式推理没有 `--target` 参数，必须传主办方提供的 anchor；输出的 I 为 anchor 原样复制，其余 11 导联来自模型：

```powershell
python scripts/predict_b2.py --checkpoint results/b2_task1_p1/b2_best.pt --task-id task1 --context-npy organizer_context.npy --context-source-type watch_ecg --anchor-npy organizer_machine_i_anchor.npy --output-dir results/b2_task1_test
```

提交前运行：`python scripts/check_b2.py`，以及下文列出的共享检查。B2 的原始输出使用固定零 baseline 作为当前合法 raw-uV 合成策略；若以后加入 baseline head，必须只使用模型预测的 baseline，不能读取真实 target baseline。

## B2 v2：P0 / P1-C1 / P1-C2 / P1-C3 baseline

本分支的当前 B2 目标是测量：严格同步的 machine-I anchor 主干之外，保守地引入跨时刻 context 是否提高 `joint_anchor_test_like` raw-V0。它不是 M1，也不包含 weak pairing、R 峰伪配对、硬时间对齐或 context-only 重建。

- `B2-P0` / `T1-P0` / `T2-P0`：Patch Transformer anchor 主干，严格 `machine-I anchor -> d12`；只读 train-only strict index。
- `B2-C1`：P0 权重初始化后，global context latent 通过 FiLM 调制 anchor token。
- `B2-C2`：gated residual adapter。
- `B2-C3`：先 FiLM，再 gated residual；FiLM 和 residual 输出零初始化，gate 初始值为 `0.05`，因此 P1 起点近似 P0。
- task1：独立 `WatchContextEncoder` 编码 watch context；仅形成全局条件。
- task2：独立 machine-d6 / body-scale-d6 encoder，各自有 canonical d6 lead embedding 和 source embedding；仅在 latent 层合并，并带 availability mask。

```powershell
# P0：严格同步 anchor 主干
python scripts/train_b2.py --experiment T1-P0 --task-id task1 --output-dir results/t1_p0
python scripts/train_b2.py --experiment T2-P0 --task-id task2 --output-dir results/t2_p0

# P1-C3：必须加载对应 P0
python scripts/train_b2.py --experiment T1-C3-watch --task-id task1 --p0-checkpoint results/t1_p0/b2_best.pt --output-dir results/t1_c3_watch
python scripts/train_b2.py --experiment T2-C3-machine --task-id task2 --p0-checkpoint results/t2_p0/b2_best.pt --output-dir results/t2_c3_machine
python scripts/train_b2.py --experiment T2-C3-body --task-id task2 --p0-checkpoint results/t2_p0/b2_best.pt --output-dir results/t2_c3_body
```

`T1-shuffle-C3-watch` 和 `T2-shuffle-C3` 仅是诊断：anchor/target 不动，只以固定 seed 将 context 换为同 split 的另一受试者。`T2-C3-both` 只有在 body/machine 同时满足相同 `subject_id`、`split`、`target_record_id`、`window_id` 的真实交集时可运行；当前适配器会写出交集计数，交集为空时 fail fast，绝不会伪造三元组。

若交集存在，machine/body/both 的公平比较使用同一交集：在 `T2-C3-machine` 或 `T2-C3-body` 后加 `--common-intersection`；`T2-C3-both` 天然只读取该交集。当前数据交集为空，因此这些共同交集实验会明确阻断。

validation 会写入 `prediction_raw.npy`、`prediction_submit.npy`、anchor、target 和 metadata；只有 submit 视图复制公开的 I anchor。正式推理不接受 target：

```powershell
python scripts/predict_b2.py --checkpoint results/t1_c3_watch/b2_best.pt --task-id task1 --anchor-npy organizer_anchor_i.npy --watch-context-npy organizer_watch.npy --output-dir results/t1_test
python scripts/predict_b2.py --checkpoint results/t2_c3_machine/b2_best.pt --task-id task2 --anchor-npy organizer_anchor_i.npy --machine-d6-context-npy organizer_machine_d6.npy --output-dir results/t2_test
```

提交前运行 `python scripts/check_b2.py`。B2 保持纯 Patch Transformer，不加载 B0 线性权重；使用 main 的 scale-aware `strict_anchor_pretrain_loss`、`joint_anchor_sync_loss` 和 raw-uV V0。P0 的 d12 scale 从 strict train index 拟合，P1 复用 P0 checkpoint 的 d12 scale；训练时不替换模型预测 I，只有 validation/test submit 组装时替换。训练使用 AdamW 和 1.0 梯度裁剪，并同时记录 centered morphology 诊断，不替代 official raw V0。

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
- 使用冻结 d12 scale 恢复到 μV 形态、并对约束残差去除窗口常数偏移后的 d12 导联代数约束；
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

不要向 main 提交原始 ECG、窗口 NPY、checkpoint、预测、患者级结果或训练日志。实验详情记录到 `docs/experiment-record-template.md`。
