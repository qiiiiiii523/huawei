# 心电图机 V6＋个人FIR25：独立候选接入说明

版本ID：`d6-context-v6-fir25-20261017`。分支：`d6-context-fir25`，基于旧`d6-context`，不覆盖旧分支，不修改B5。不是V7/V8，也不是旧V6原输出。

## 给队长的建议

先将**独立后处理**接到现有B5输出后做对照，不必立刻替换主干。若要复现本文绝对成绩，再使用本包专门保存的V6参考EMA权重及其冻结预处理。

后处理可接任意合法的当前12导联预测，但在新B5-best上是否有效仍要复验，不能把本包结果直接当作新主干结果。只处理心电图机历史context，体脂秤/手表来源未经本包验证，不要默认套用。

## 方法与接口

1. 六导联历史实际近二维；仅用历史I、II及有效mask拟合个人I→II短FIR。
2. 固定61点、500Hz、alpha=0.01，完整历史拟合；历史前40%/后40%隔离检查仅用于可用性，不按分数门槛调混合权重。
3. 从五个肢体预测等权估计一个与**当前观测I**相容的II，即一致性投影。
4. 投影II与个人FIR估计II按75%/25%混合，再生成II/III/aVR/aVL/aVF；V1–V6和预测I不变。
5. 历史缺失、过短、无足够有效点、平直或退化：关闭FIR，回退**V6/B5输出的一致性投影**，不是旧实验的公共滤波器，也不是完全未经投影的原输出。

独立后处理仅依赖NumPy；不会加载患者查表系数、人口学或当前真实II。个人系数必须从该次测试合法提供的历史重新计算；每条记录计算一次，不逐窗拟合。

```python
from baselines.context_d6_fir_postprocess import historical_fir, apply_postprocess

# 所有波形必须是500Hz、原始物理单位μV，不能输入逐导联归一化张量。
# history_uV: [6, T_hist]，顺序I,II,III,aVR,aVL,aVF；不与当前记录时间对齐。
# history_mask: 同shape布尔数组，合并有限点、缺口、尾部padding和输入质量mask。
# 无历史传None；不得插值补造无效历史。
fir = historical_fir(history_uV, history_mask)

# v6_or_b5_prediction_uV: [12,T_now]；current_observed_i_uV: [1,T_now]或[T_now]。
# 当前目标按时间顺序拼完整120秒，再做FIR/混合，避免逐10秒窗口边界效应。
enhanced, status = apply_postprocess(
    v6_or_b5_prediction_uV, current_observed_i_uV, fir, strength=0.25
)
```

标准12导联顺序必须为`I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6`。若主干只输出11条，先在最前面拼接可见当前I，不能猜测通道顺序。不要用模型预测I代替观测I参与约束。输入为mV时应乘1000再调用；scale还原和导联顺序由调用方明确保证，模块无法自动识别错误单位。

必须先把原模型输出逆变换回μV，再处理；**历史也要读原始μV**，不能只乘scale而忘记历史median。旧context缓存`raw_uV`可直接使用。历史与当前长度无需相同。FIR使用当前I局部前后邻域，是离线非因果算法，不声称实时因果。

## 完整V6参考快照（可选，不必替换队长主干）

权重：`artifacts/v6_fir25_20261017/ema_inference.pt`，由本地exp013的seed42参考EMA精简得到，去掉优化器、训练RNG等；保留模型结构、context配置、EMA和冻结scales，不含患者ECG或个人滤波器查表。

**不要误用旧路径**`artifacts/v6_reliability_gate/context_seed42/best.pt`：那是先前已发布V6的另一份权重；本包绝对结果对应本新快照。原文件保持不动。

```python
from baselines.context_d6_fir_release import load_release
model, preprocessor = load_release(
    'artifacts/v6_fir25_20261017/ema_inference.pt', device='cuda'
)
```

后续沿用本分支V6的record-context collator、当前I/人口学条件及Flow Matching采样流程（EMA/Heun16/K16/seed42）。loader不自动做采样或FIR，也不需要原B5训练checkpoint；预处理配置仍为仓库`configs/preprocessing.yaml`。只加载`ema`，不可把推理包用于原地续训。V6继承V2–V5，这些依赖已由基分支提供，不需要引入本地V7/V8。

## 结果与限制

机器组缺失11导联r，40验证患者，120秒逐记录评分：

| 权重 | 原V6 | 一致性投影 | 25%个人FIR，投影安全回退 |
|---|---:|---:|---:|
| seed42参考（本包） | 0.622310 | 0.622564 | 0.628074 |
| seed123复验 | 0.619565 | 0.619817 | 0.625916 |
| seed2026复验 | 0.618557 | 0.618813 | 0.625094 |

独立GPU端到端重推本包同一权重，原V6 r=0.622299、安全增强r=0.628061；跨环境神经推理不是逐点一致，但增益复现。两种子原训练选模为Heun16/K16、seed42参考为Heun4/K1，不能包装为严格同协议三种子平均。

本人历史优于公共及三个打乱对照，支持额外个人信息。收益只来自肢体导联，胸导联完全不变。这里不是两设备Task2或两任务比赛最终平均分。

**风险：**安全回退后仍有12～15/40位患者下降，最差患者整体r下降约0.024、单导联约0.13。它是平均r增强候选，不是无损或临床安全保证。独立患者测试尚缺失；不要按当前真实II挑患者权重或继续细搜比例。若新主干/新QC/新患者不改善，可关闭FIR并单独对照投影/原输出。

## 随包检查

```powershell
python -m pytest -q baselines/context_d6_v2/tests/test_fir_postprocess.py
```

覆盖7项用例（含5种无效历史）、输入不变、胸导联不变、已知FIR和缺口屏蔽。独立封装后重新读取原始合法历史及观测I，验证结果仍为0.628074，系数和有效历史后处理波形与实验完全一致。模型EMA张量和冻结scales与来源快照逐项一致，详见同目录manifest.json。

不上传：患者原始数据、人口学、逐患者报告、私人FIR系数、完整预测、失败版本、V7/V8、本地服务器凭据。队长在自有合法数据上运行原有B5输入流程即可。
