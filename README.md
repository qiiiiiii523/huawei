# 12 导联 ECG 信号生成项目

本仓库的 main 分支只维护公共数据契约、实验协议、预处理、评价和检查工具，不包含 B0/B1/B2/M1 的具体网络，也不在此分支启动正式训练。

## 一、仓库目录

```text
huawei_upload_tmp/
├── configs/                 # 公共配置和 A/B/C 实验配置
├── metadata/                # subject split、配对清单、严格 d12 索引
├── ecg12gen/                # 数据集、预处理、loss、评价等公共代码
├── scripts/                 # 构建和只读检查脚本
├── docs/                    # 数据治理和实验记录模板
├── reports/                 # 已有 QC/配对/划分报告
├── README.md
└── .gitignore
```

关键文件：

- configs/common.yaml：路径、采样率、窗口、导联顺序、seed 和数据变体。
- configs/preprocessing.yaml：统一预处理 v1。
- configs/training_protocol_v1.yaml：固定 train/validation、loss 使用边界和验证规则。
- configs/experiments/：A/B/C 三条训练路线的配置。
- metadata/subject_split.csv：唯一允许使用的 subject-level 划分。
- metadata/d12_strict_pretrain_index.csv：去重后的 train-only d12 预训练索引。
- ecg12gen/dataset.py：task1/task2 跨设备弱配对读取器。
- ecg12gen/d12_pretrain.py：严格 d12-I/d12-six 预训练读取器。
- ecg12gen/body_scale.py：体脂秤 A/B 输入读取器。
- ecg12gen/preprocessing.py：非破坏性的 train-fitted 预处理。
- ecg12gen/evaluate.py：官方 V0 和 centered 形态诊断。

切好的数组不放进 Git 仓库，目录与仓库同级：

```text
HW/
├── huawei_upload_tmp/
├── task1_output/                  # task1 input [N,1,5000]、target [N,12,5000]
├── task2_output/                  # task2 input [N,6,5000]、target [N,12,5000]
├── task2_body_scale_ablation/     # 体脂秤 B 版本输入
└── task1_rpeak_pseudo_output/     # task1 C 路线派生数据，可选
```

## 二、已固定的 v0.1 实验协议

- ECG 单位：μV；采样率：500 Hz。
- 每个窗口 10 秒、5000 点；步长 10 秒，不重叠。
- d12 顺序：I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6。
- d6 顺序：I、II、III、aVR、aVL、aVF。
- train/validation 按 subject 划分，固定为 88/22（约 80%/20%），seed=42。
- 同一 subject 的所有设备、记录和窗口必须在同一 split。
- 短于 30 秒的记录保留审计，但不进入默认训练。
- 默认训练只允许 pair_status=paired 且输入、target 为 usable 的样本。
- review、unmatched、reject 不自动进入训练。
- PPG 和 acceleration 保持 100 Hz，第一版不作为主输入。
- 原始数据和切窗后的 NPY 不修改、不重切。

## 三、三条训练路线

| 路线 | task1 | task2 |
|---|---|---|
| A | watch 单导联 → d12 弱配对适配 | 心电机/体脂秤 d6 → d12 弱配对适配 |
| B | d12-I 严格预训练 → 弱配对微调 | d12-six 严格预训练 → 弱配对微调 |
| C | d12-I 严格预训练 → R 峰伪配对微调 | 无 C 路线 |

严格 d12 预训练只读取 d12_strict_pretrain_index.csv：

- d12-I：Y_12lead[0:1] → Y_12lead[0:12]；
- d12-six：Y_12lead[0:6] → Y_12lead[0:12]；
- 只使用 train，索引按 target_record_id 和窗口采样范围去重。

R 峰伪配对只用于 task1 C 路线。当前已生成 task1_rpeak_pseudo_output/；task2 不生成 R 峰伪配对数据，也不设置 C 路线。

## 四、窗口对齐和体脂秤 A/B

- d12 内部预训练：输入和 target 是同一 d12 记录、同一窗口，允许逐点 loss。
- 普通跨设备配对：只依据可靠 subject_id 配对，不假设逐点同步；默认禁止逐点 target MSE。
- 体脂秤 A：原始连续记录直接切 10 秒窗口。
- 体脂秤 B：连续记录先做 0.2 Hz 去趋势，再切 10 秒窗口。
- A/B 使用相同 subject split 和相同 task2 target；B 输入通过 canonical_array_index 对齐 target。
- A/B 必须作为独立实验版本比较，不要无记录混合。

体脂秤读取示例：

```python
from ecg12gen.body_scale import BodyScaleVariantDataset
from ecg12gen.dataset import ECGDataConfig
config = ECGDataConfig.from_yaml("configs/common.yaml")
train_a = BodyScaleVariantDataset(config, "train", "A_raw_window")
train_b = BodyScaleVariantDataset(config, "train", "B_detrend_0p2Hz_then_window")
```

## 五、统一数据预处理

所有 ECG 输入和 d12 target 都要调用 main 的 ECGPreprocessor：

1. 只用 train 拟合每个设备类型和 d12 的 scale；
2. 每窗口、每导联减去 median；
3. 使用 train 拟合并冻结的 scale；
4. train、validation、推理使用同一冻结实例；
5. 不改写原始 NPY。

体脂秤 B 的 0.2 Hz 去趋势只是输入变体，之后仍然执行上述共享预处理。预处理后的模型视图是 centered/scaled ECG；不得用真实 validation target 的 baseline 恢复预测。

## 六、loss 和评价

- 同步 d12：缺失导联 Huber/PCC + observed consistency + 生理约束。
- 普通弱配对：observed consistency + 独立 strict-train d12 分布约束 + 生理约束，默认不做逐点 target MSE。
- R 峰伪配对：只使用 accepted 样本，并按 alignment_quality_score 加权。
- 初始权重在 configs/losses.yaml，权重校准只能使用 train。

官方 V0 由 task1/task2 共用：

- task1：12 导联平均 Pearson r，得到 r1；
- task2：12 导联平均 Pearson r，得到 r2；
- task2：缺失 V1–V6 平均 RMSE；
- 主分数：0.5 × r1 + 0.5 × r2；
- centered diagnostic 只用于形态诊断，不替代官方 raw V0。

评价命令：

```powershell
python -m ecg12gen.evaluate --prediction results/task1_validation_prediction.npy --target ../task1_output/task1_validation_target.npy --metadata ../task1_output/task1_window_metadata.csv --task-id task1 --output-dir results/task1
```

每次实验保存 overall_metrics.csv、lead_metrics.csv、report.md，并记录配置、输入变体、loss、baseline head/adapter 和 validation 结果。

## 七、B0/B1/B2 公平比较规则

B0、B1、B2 首轮横向比较必须使用相同的 task、输入版本、subject split、预处理、seed 和训练设置：seed=42、deterministic=true、batch size=16、每个阶段最多 100 epochs、AdamW、learning rate=0.001、weight decay=0.0001、无 scheduler、无 gradient clipping、关闭 early stopping。checkpoint 统一选择 validation 官方 V0 最佳结果；指标相同时取较早 epoch。

模型结构可以不同，但不能同时改变数据路线、输入变体或上述训练预算，否则结果不能直接归因于网络结构。M1 及后续消融可以覆盖这些默认设置，但必须在实验记录中写明。

## 八、模型分工与 adapter 规则

建议的 Git 分支：

```text
main
├── baseline/B0
├── baseline/B1
├── baseline/B2
└── candidate/M1
```

- B0/B1/B2：只实现各自网络，第一轮不使用 adapter 和 baseline head。
- M1：先做 M1-no-adapter；只有设备分层 centered 指标明显变差时，才加入 device adapter。
- 如果 centered 指标好但 raw 指标差，再单独验证 baseline head。
- baseline head 不是 B0 模型，而是预测 d12 每导联窗口 baseline 的可选输出头。
- main 只提供 raw 合成接口，不实现具体 baseline head。

## 九、组员使用 main 的固定流程

1. 从已确认的 main 创建分支：

```powershell
git switch main
git pull --ff-only
git switch -c baseline/B0
```

2. 按本文目录准备受控数据，运行检查：

```powershell
python scripts/check_d0_d1_v0.py --config configs/common.yaml
python scripts/check_experiment_protocol.py
python scripts/check_preprocessing_protocol.py
python scripts/check_abc_experiment_protocol.py
python scripts/check_body_scale_variants.py
```

3. 选择对应 Dataset 和 A/B/C 配置；只用 train 拟合 ECGPreprocessor。
4. 模型输入必须保留 X_ecg、lead_mask、missing_mask、task_id、supervision_mode、alignment_mode 等字段。
5. 按监督类型使用对应 loss，不能把弱配对当作同步波形。
6. 只在 validation 上运行 V0，不能用 validation target 训练或调权。
7. 在分支实验记录中写明网络、输入版本、loss、baseline head/adapter 和 V0 结果。

禁止提交原始 ECG、task1_output、task2_output、task2_body_scale_ablation、task1_rpeak_pseudo_output、checkpoint 和敏感数据。
