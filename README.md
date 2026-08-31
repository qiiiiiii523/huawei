# 12 导联 ECG 信号生成技术

本仓库是比赛项目的 `main` 主分支公共基础。它固定 D0/D1/V0 数据与评估协议、受试者级数据划分、公共训练参数、随机种子和实验记录格式；不包含 B0/B1/B2 模型，也不会在此阶段启动正式训练。

## 仓库目录说明

```text
huawei_upload_tmp/
├─ configs/                 # 统一数据、质量和训练协议配置
├─ metadata/                # 固定划分、配对清单和字段映射
├─ reports/                 # 数据盘点、QC、配对和划分报告
├─ docs/                    # 数据治理说明和实验记录模板
├─ ecg12gen/                # D0/D1 数据接口、V0 评价、训练公共工具
├─ scripts/                 # 只读数据/协议检查脚本
├─ README.md                # 本说明
├─ requirements.txt         # 最小 Python 依赖
└─ .gitignore               # 大型数据、结果和本地环境排除规则
```

目录用途：

- `configs/common.yaml`：所有路径均为相对路径，定义 ECG/PPG/acc 采样率、窗口、导联顺序、D0 默认任务、seed 和输出位置。
- `configs/data_spec.yaml`：数据单位、导联顺序、窗口和模态的统一规范。
- `configs/quality_rules.yaml`：`usable/review/reject` QC 规则；短于 30 秒记录保留审计但不进入 v0.1 默认训练。
- `configs/training.yaml`：模型无关的公共训练参数、固定划分引用、验证规则和结果格式。
- `metadata/subject_split.csv`：唯一允许使用的正式 train/validation 划分。
- `metadata/pair_manifest_task1.csv`、`metadata/pair_manifest_task2.csv`：跨设备可靠配对依据。
- `reports/`：已有的盘点、质量、配对和划分结果，供审计而非训练读取。
- `docs/data-governance.md`：原始数据保护、配对和划分的详细约束。
- `docs/experiment-record-template.md`：每次 B0/B1/B2 实验应复制使用的记录格式。
- `ecg12gen/contracts.py`、`dataset.py`：D0/D1 统一接口与监督门控。
- `ecg12gen/evaluate.py`：V0 统一验证指标和 CSV/Markdown 报告。
- `ecg12gen/training.py`：公共 seed 和训练协议读取工具，不包含模型或训练循环。

`task1_output/`、`task2_output/`、原始 `Data/`、模型权重和生成结果都在仓库外或被 `.gitignore` 排除，不能上传 GitHub。代码通过 `configs/common.yaml` 的相对路径读取本地处理结果。

## main 已固定的 v0.1 实验协议

| 项目 | 当前固定值 | 位置 |
|---|---|---|
| ECG 单位 | μV | `configs/common.yaml`、`configs/data_spec.yaml` |
| ECG 模型采样率 | 500 Hz | 同上 |
| d12 导联顺序 | I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6 | 同上 |
| d6 导联顺序 | I、II、III、aVR、aVL、aVF | 同上 |
| 窗口 / 步长 | 10 秒 / 10 秒，不重叠 | `data_spec.yaml`、`common.yaml` |
| 训练 / 验证划分 | 固定受试者级 88 / 22，即 80% / 20% | `metadata/subject_split.csv` |
| 划分 seed | 42 | `subject_split.csv`、`training.yaml` |
| 默认训练质量 | `paired` + 输入/目标均 `usable` | D1 和 `quality_rules.yaml` |
| 少于 30 秒记录 | 保留审计，不作为默认训练数据 | `quality_rules.yaml` |
| PPG | 保持 100 Hz，第一版不作为主输入 | `data_spec.yaml`、`common.yaml` |
| acceleration | 保持 100 Hz，第一版不作为主输入 | `common.yaml` |
| 标准化 | 当前为 `none`；启用时必须训练/验证一致并记录 | `training.yaml` |
| 公共 seed | 42、确定性模式 | `training.yaml`、`training.py` |
| 结果记录 | CSV/Markdown V0 报告 + 实验记录模板 | `evaluate.py`、`docs/experiment-record-template.md` |

因此，针对你列出的参数要求，当前均符合。比赛正式指标已固定：r1/r2 均为 12 个导联 Pearson r 的非加权平均；主分为 `0.5 × r1 + 0.5 × r2`；任务二缺失导联平均 RMSE ≤ 70 μV 时加 10 分，否则加 `700 / RMSE` 分。

## D0：统一数据接口

`UnifiedECGDataset` 使用队友已经生成的 NPY 窗口，不重新解析 XML/JSON/CSV，也不修改原始数据。每个样本提供：

- `X_ecg`：实际输入 ECG；任务一 `[1, 5000]`，任务二 `[6, 5000]`。
- `lead_mask` / `missing_mask`：在标准 12 导联顺序下标记已提供和缺失导联。
- `Y_12lead`：完整目标，形状 `[12, 5000]`。
- `task_id`、`split`、`meta`、可选 `ppg` / `acc` 与 `modality_mask`。
- `supervision_mode`、`pairing_type`、`alignment_mode`、`pair_confidence`、`pair_status`。

d12 预训练不生成额外文件：`d12_i_pretrain` 动态取 `Y_12lead[:1]`，`d12_six_pretrain` 动态取 `Y_12lead[:6]`；两种模式只允许 `train`。

## D1：两类监督严格区分

- d12 内部同步导联重构：输入与目标来自同一条 d12 记录，供 d12-I / d12 六导联预训练使用，仅用训练集目标。
- 跨设备弱配对：手表 ECG、心电图机 d6 或体脂秤 d6 与 d12 只按可靠 `subject_id` 配对。接口将其标为 `weak_subject_pair_record_start`，并设置 `pointwise_mse_allowed=False`；不能默认把它当作逐点同步波形 MSE。

默认读取器只返回 `pair_status=paired` 且输入、目标和窗口均为 `usable` 的记录。`review` 与 `unmatched` 不会自动进入训练。

## V0：统一验证协议

V0 的 `evaluate_predictions` 由 task1/task2 共用：逐导联汇总所有 validation 窗口的采样点，计算 Pearson r 与 RMSE；12 个 r 的非加权平均就是任务一 r1 或任务二 r2。任务二默认缺失导联为 V1～V6，其逐导联 RMSE 的平均值用于加分。

### 任务二补充诊断（不参与正式计分）

任务二的 12 导联总体指标仍保留，以保持与正式 `task2_r2` 一致；但已有 I、II、III、aVR、aVL、aVF 六个肢体导联可被模型直接复制，不能单独用它判断生成质量。因此 task2 评测还会额外报告真正需要生成的 V1～V6 的平均 Pearson r 与平均 RMSE。

为避免窗口较多的受试者主导分析，补充报告会先将同一受试者的全部窗口及采样点按导联聚合，计算该受试者的指标，再对受试者等权平均（`subject_macro`）。同时按 `input_type` 分开输出 `ecg_machine_d6` 和 `body_scale_d6` 的：

- `pooled_windows`：该设备全部窗口的点级汇总，与正式 V0 的窗口加权方式一致；
- `subject_macro`：该设备内每位受试者等权的结果。

上述诊断只用于识别设备域差异和真实胸前导联生成能力；不会替代 `task2_r2`，也不会改变比赛主分或 RMSE 加分。

双任务完成后，`competition_score` 计算：

- 主分：`0.5 × r1 + 0.5 × r2`；
- 加分：任务二缺失导联平均 RMSE ≤ 70 μV 时为 10 分，否则为 `700 / RMSE` 分；
- 比赛总分：主分 + 加分。

每项任务都会生成基础 V0 报告：

- `overall_metrics.csv`
- `lead_metrics.csv`
- `report.md`

任务二还会自动生成补充诊断报告：

- `task2_subject_metrics.csv`：每位受试者先聚合窗口后的 12 导联与 V1～V6 指标；
- `task2_device_metrics.csv`：`ecg_machine_d6`、`body_scale_d6` 的 `pooled_windows` 与 `subject_macro` 指标，以及总体的受试者宏平均。

```powershell
python -m ecg12gen.evaluate `
  --prediction results/task1_validation_prediction.npy `
  --target ../task1_output/task1_validation_target.npy `
  --metadata ../task1_output/task1_window_metadata.csv `
  --task-id task1 --output-dir results/task1
```

完成两个任务评估后汇总总分：

```powershell
python scripts/summarize_competition_score.py `
  --task1-overall results/task1/overall_metrics.csv `
  --task2-overall results/task2/overall_metrics.csv `
  --output-dir results/competition
```

## 检查与后续实验

```powershell
python scripts/check_d0_d1_v0.py --config configs/common.yaml
python scripts/check_experiment_protocol.py
```

两项检查都只读取数据和协议，不训练模型。开始 B0/B1/B2 前，每次实验应使用 `docs/experiment-record-template.md`，并仅用 V0 validation 结果做模型比较。
## 预处理提案（待团队确认）

`configs/preprocessing.yaml` 和 `ecg12gen/preprocessing.py` 定义非破坏性的模型输入预处理：不同设备输入分别使用 `T_watch`、`T_machine_d6`、`T_body_scale_d6`；所有任务的 d12 target 均使用同一个 `T_d12`。变换在读取时动态执行，原始 NPY 不会被改写；尺度仅由训练集拟合，validation/test 只应用冻结统计量。该提案尚未作为正式实验协议启用。
## 预处理如何工作（待团队确认）

预处理不会改写、覆盖或另存现有的 `.npy`。原始数组始终保留为 `raw ECG`；模型训练时才在内存中动态调用预处理，得到新的模型视图。

```text
raw ECG window
  → ECGPreprocessor.transform_window(...)
  → model_signal（去基线、缩放、裁剪后的新数组）
  → B0 / B1 / B2 / M1 模型
```

每次变换同时返回 `baseline_uV`、`scale_uV` 和 `source_type`，便于审计；这些返回值不回写原始 ECG。当前 `UnifiedECGDataset` 仍默认返回 raw ECG，预处理提案尚未自动启用，后续训练入口必须显式按如下顺序调用：

1. 只用固定 `train` split 的输入设备数据和 d12 target 调用 `ECGPreprocessor.fit(...)`，得到冻结的尺度统计量；
2. train、validation 和推理都只应用同一个冻结 preprocessor；
3. 输入按设备类型调用 `T_watch`、`T_machine_d6` 或 `T_body_scale_d6`；
4. 无论任务一、任务二或 d12 内部预训练，d12 target 都调用同一个 `T_d12`。

`configs/preprocessing.yaml` 的字段含义：

| 字段 | 当前提案值 | 含义 |
|---|---|---|
| `raw_data_mutation` | `false` | 禁止改写原始数组 |
| `baseline.method` | `per_window_per_lead_median` | 每窗口、每导联减去 robust median，处理 DC/慢基线位置 |
| `scaling.method` | `train_median_p5_p95_range` | 每设备、每导联以训练集窗口 P95-P5 动态幅度中位数作为尺度 |
| `scaling.fit_split` | `train` | 明确禁止用 validation 拟合尺度 |
| `minimum_scale_uV` | `25.0` | 防止近乎平坦导联除以极小数 |
| `clip_model_signal` | `12.0` | 防止少数异常点主导数值范围 |
| `sources` | watch / machine d6 / body d6 / d12 | 各来源的预期导联数 |
| `target_transform` | `d12` | 所有任务的 d12 target 共用同一变换 |

`ecg12gen/preprocessing.py` 是实际可被训练代码调用的预处理库；`scripts/check_preprocessing_proposal.py` 只是合成数据合同测试，不会处理真实数据、不产生训练数据，也不改变任何配置。它目前验证：尺度只能由训练数据拟合、体脂秤 DC 能在模型视图中居中、raw 数组不变，以及 d12 target 使用统一变换。
