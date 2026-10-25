# FM+Unet：七层慢 U-Net 与原三层基准

## 当前代码：七个分辨率层级

本次只扩充慢分支，固定 B5-fast 及快慢相加方式不变。新建 `build()` 默认使用 5Hz、120秒输入/输出的七层 U-Net：

| 层级 | 通道数 | 120秒的时间点数 |
|---|---:|---:|
| E0 | 64 | 600 |
| E1 | 128 | 300 |
| E2 | 256 | 150 |
| E3 | 256 | 75 |
| E4 | 256 | 37 |
| E5 | 256 | 18 |
| 瓶颈 | 256 | 9 |

共6次stride=2下采样，解码端6次上采样，并与相同层级的编码特征拼接。上采样显式对齐跳连长度，因此37/75等奇数长度可正确恢复。采用M7宽度，高层通道封顶256。每个编码/解码层级2个残差块，瓶颈6个膨胀残差块（d=1,2,4,8,16,32）；共30个残差块、14,506,847个参数。FM时间和人口学嵌入均为128维，注入全部残差块。前一版窄七层是3,064,687参数，现已扩宽。

```python
from FM_Unet.fm_experiment import build
slow_model = build()  # 新七层模型，随机初始化，需训练
assert len(slow_model.config.channels) == 7
# 保存新权重时必须一并保存 slow_model.config.to_dict() 到 model_config。
```

**尚未训练七层模型，以下分数全部属于原三层模型。** 仓库的 `best.pt` 保留原始三层EMA权重；`SlowPredictor()` 按checkpoint中的 `model_config` 严格重建，因此不传新checkpoint时仍运行三层基准。训练好七层模型后，通过 `SlowPredictor(checkpoint="七层模型路径", device="cuda")` 加载；checkpoint需包含与原接口一致的 `ema`、`model_config`、`native_hz`、`input_seconds`、`output_seconds`、`scale_uV`、`noise_std7`。旧三层权重不能直接作为七层训练成果。本分支现已补齐 `python -m FM_Unet.train` 训练入口，支持三阶段串行、恢复与最终验证。

七层输入至少需要64个native时间点；5Hz的真实10秒只有50点，会明确报错。短记录预训练需另选可支持的分辨率（如本次配置的10Hz真实10秒），不可直接沿用5Hz的10秒输入。`build(rate=10/50)` 仍可显式选择其他分辨率。

验证：`python -m unittest discover -s FM_Unet/tests -v`，覆盖七层前向/反向、奇数长度恢复、5/10/50Hz路径、短输入检查、采样输出、配置序列化及旧checkpoint严格加载和前向逐值一致性。

真实batch64训练可运行 `bash scripts/run_fm_unet_m7.sh`（服务器默认路径，物理GPU2）。中途不验证，每阶段仅最终EMA做Heun16/K16验证；三个阶段预算2000/4000/6000。华为最终以固定快+新慢完整波形评分。训练后的慢分支部署文件为 `huawei120/inference.pt`。

实验计划、优化器与训练注意事项见 [七层M7宽度实验配置](../docs/fm_unet_m7_training.md)，配置文件在 `configs/fm_unet_m7/`。配置检查可运行 `python -m FM_Unet.experiment --config configs/fm_unet_m7/huawei120.json`；这不是训练命令。

## 原三层基准（随附best.pt）

5Hz、七通道、120秒输入→120秒输出，365231参数。原始I保留500Hz，不使用I独立多尺度编码器。FM采用Heun16、K16正负噪声配对平均，固定种子42并结合record_key。

## 接入
将整个 `FM_Unet` 文件夹放到主项目根目录，只依赖PyTorch和NumPy。`best.pt`仅含最佳EMA权重、模型配置及冻结scale/noise，不含患者数据、优化器或缓存。

```python
from FM_Unet import SlowPredictor
import numpy as np
slow_model = SlowPredictor(device="cuda")
# raw_i_uV: 120秒原始I，500Hz，60000点，单位微伏。
# metadata: 原B5人口学编码，无batch维；numeric[3]、sex标量、field_mask[4]、age_topcoded[1]。
slow11 = slow_model.predict(raw_i_uV, metadata, record_key="stable_record_id")
# fast11必须是原B5-fast固定快残差，不能是完整FM波形。
combined11 = (fast11 + slow11).astype(np.float32)
full12 = np.concatenate([np.asarray(raw_i_uV).reshape(1,60000), combined11], axis=0)
```

返回顺序：II、III、aVR、aVL、aVF、V1–V6。直接预测II＋V1–V6，其余肢体慢分量用MA501(I)及预测II推导。不要再乘旧P的0.75、不要重复加G120/context/旧慢CNN。保持record_key稳定以复现噪声。

快分量必须使用B5-fast e264be0对应FM_P_best45 EMA：完整120秒生成波形减去全记录MA501，edge padding；Heun16、K16。快权重SHA256：961aad3b8212bef020103de9be02471a50bb5d00905ec9e6ee2f39aefb919bc1。本包不重复上传已有固定快权重。

## 训练及结果
PTB-XL＋Chapman真实10秒2000更新 → PTB Diagnostic真实75秒4000更新 → 华为120秒6000更新，最佳华为step6000。华为156训练患者、39可靠验证患者/记录；原患者划分、QC及120秒评分。

| 分量 | 肢体5导联r | 胸6导联r | 全11导联r |
|---|---:|---:|---:|
| 固定快 |0.886722|0.893897|0.890636|
| 新慢 |0.412366|0.413099|0.412765|
| 快＋慢 |0.794493|0.671698|0.727514|

同一固定快的P＋G120参考0.725310，提升0.002203。以上是反复使用的本地验证集、单种子结果，不是赛方成绩或独立测试。快/慢r仅诊断；选模用完整波形120秒、排除I、原QC、Task2设备等权的原评分。

原始I先每10点整理为10通道，与7通道状态拼接，在50Hz输入卷积后降至5Hz。U-Net通道16/32/64、2次下采样、17个残差块。人口学及FM时间注入残差块。输入需已按原B5规范预处理，勿自行改变单位、导联顺序或scale。
