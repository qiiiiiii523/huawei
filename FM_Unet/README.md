# FM+Unet：当前最佳慢分支

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
