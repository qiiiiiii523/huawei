# B5 条件 Flow Matching：完整网络、条件融合与实验方案

版本：v2，更新日期：2026-10-08。

本文替换此前设计稿。数据、预处理和评分以当前本地 C:/Users/Ashley/Desktop/HW/huawei 的 main 代码、配置及 2026-10-07 更新说明为准，不使用历史实验结果或其他预处理口径。B5-U 基础主线已提供网络、元信息、公开数据适配、训练/微调、采样和预测代码；历史 ECG/PPG 与 CNN＋Transformer 仍为后续扩展方案。没有启动训练；文中的预期收益均需验证。

## 1. 直接确定的主方案

**B5 主模型：条件一维 U-Net 作为速度网络，采用线性路径 Conditional Flow Matching；同步 I 和人口学条件从公开预训练开始参与训练，随后在华为数据上微调。历史 ECG、PPG 是独立扩展版本的新增条件。**

第一版主条件固定为：

\[
c_{\mathrm{base}}=\{I_B,\mathrm{age},\mathrm{sex},\mathrm{height},\mathrm{weight},\mathrm{field\ masks}\}.
\]

同步 I 与目标同记录、同时间窗。人口学字段允许缺失；同步 I 不作为可缺失模态。最终研究目标是恢复 B 时刻 II、III、aVR、aVL、aVF、V1–V6，正式评估只统计这 11 导联。

扩展版本的条件为：

\[
c_{\mathrm{ext}}=\{c_{\mathrm{base}},E_{\mathrm{ECG}}(X_A),E_{\mathrm{PPG}}(P_A),
\mathrm{modality\ masks}\}.
\]

A 时刻的历史 ECG/PPG 不与 B 时刻目标逐点对齐。

这条主线使用自己搭建并从随机初始化训练的 B5，不要求使用别人的预训练权重，也不直接继承 B4 的 diffusion 权重。公开预训练得到的 checkpoint 是自己的模型；华为微调是继续训练这个模型。

## 2. 网络、框架、条件与融合的区别

| 层次 | 定义 | 本方案选择 |
|---|---|---|
| 训练与生成框架 | 如何构造训练状态、学习何种输出、怎样生成最终波形 | Conditional Flow Matching，线性插值路径，ODE 采样 |
| 网络骨干 | 可学习函数如何提取、组织和恢复特征 | 第一版 1D U-Net；后续另做 CNN＋Transformer 对照 |
| 条件编码器 | 把不同输入变成可融合的特征 | 同步 I 多尺度 CNN，人口学 MLP，历史 ECG/PPG 小 CNN |
| 条件融合 | 条件如何影响速度网络 | 同步 I 特征投影相加；人口学 FiLM；历史模态门控 FiLM |
| 数据阶段 | 权重在哪种数据上学习 | 公开从零预训练 → 华为微调 → 华为历史条件扩展 |

U-Net 在这里主要由 CNN 构成，并通过下采样、上采样和跳连组织。Transformer 可以加入 U-Net；它不与 FM 构成二选一。

B4 与 B5 可以采用相近的 U-Net 结构，但 diffusion 的路径、监督目标和采样规则不同于本方案 FM。复用骨干代码不等于复用训练好的权重，更不等于只改模型名称。比较 FM 与 diffusion 时，先控制骨干、条件、数据与训练预算；比较骨干时，再固定 FM。

## 3. “条件固定”和分阶段新增条件

条件固定有三种不同含义：

1. 同一实验的输入 schema 固定，例如公开预训练和华为基础微调始终使用同步 I＋四个人口学字段及 mask。
2. 对某条记录做一次 ODE 生成时，条件的值在整个积分过程中固定；变化的是 x_t 和 t。更换患者后条件值会变化。
3. 条件编码器的参数可以训练、微调，不需要永久冻结。

第一版确定使用的人口学编码器，从随机初始化的公开预训练开始学习。后续加入历史模态时，增加独立编码器与注入模块，继承已学好的基础主干；这是新版本、新实验，不是在同一个实验中随意改变条件定义。

建议为基础网络预留条件注入接口，但无须让没有训练数据的 PPG 编码器在公开阶段运行。新增模块必须明确出现在 checkpoint 加载审计中，不能用无检查的宽松加载掩盖主干不匹配。

注意命名：main 中 P0_anchor_only 是严格 I-only。人口学版本命名为 B5-meta，不把它冒称为未修改的公共 P0。历史扩展阶段可参考 main 的 P1 关系，但新增 B5/PPG 需要单独配置；当前 context_fusion_protocol 的适用模型列表未包含 B5，也未包含 PPG。

## 4. 当前数据与预处理

### 4.1 本地数据事实

| 项目 | 已核查结果 | 实验影响 |
|---|---|---|
| 患者划分 | 110 人，88 train / 22 validation | 所有设备沿用现有患者划分 |
| 同步 train-only 索引 | 938 窗，78 人/78 记录 | 基础华为训练去重使用当前索引 |
| 用户信息 | 103 行、101 个 externalid | 按统一 subject_id 关联，不按行序 |
| 元信息覆盖 | 覆盖同步训练的 71/78 人 | 缺失样本保留，使用字段 mask |
| 年龄/性别 | 用户信息表年龄 20–28 岁，全部男性 | 年龄、性别收益不能预设；不代表隐藏测试分布 |
| 身高/体重 | 字段存在；有一组重复患者人口学信息冲突 | 核对单位，冲突字段按可审计规则处理 |
| 病史 | 用户信息表未发现该字段 | 首版不作为条件 |
| PPG 索引 | 131 条，130 条可与手表按 subject/group 一一匹配 | 索引匹配不等于波形可用或与 d12 同步 |

PPG 既有 QC 报告记录 usable 87、review 44；其原生单位为 raw counts、采样率 100 Hz。这是已有索引/QC 结果，不代表本方案已重新验证全部原始 PPG 波形。窗口级 PPG 质量和缺口还需要在实施时检查。

### 4.2 ECG 数值表示：严格沿用 main

预处理版本为 v3.0-raw-target-record-context。

| 信号角色 | 模型空间变换 | 减 median？ |
|---|---|---|
| 同步 anchor I，ecg_machine_i | raw_I / frozen_d12_I_scale | 否 |
| d12 target | raw_d12 / frozen_d12_scale | 否 |
| 历史 watch_ecg | (raw_watch − record_median) / frozen_watch_scale | 是，按记录 |
| 历史 ecg_machine_d6 | (raw_d6 − record_median_per_lead) / frozen_machine_d6_scale | 是，按记录 |
| 历史 body_scale_d6 | (raw_body − record_median_per_lead) / frozen_body_scale | 是，按记录 |

**同步心电图机 I 与 d12 target 不减 median；心电图机 d6 作为历史 context 时，当前代码也减记录 median。**

scale 仅由训练数据拟合：各窗口 P95−P5，再对窗口取中位数，下限 25 μV。anchor I 使用 d12 I 的同一 scale。所有阶段冻结并复用数值表示。

context 不逐窗口重新计算 median。Task1 使用存储的物理记录 median；Task2 当前从同记录缓存窗口去重统计，覆盖缓存保留部分。体脂秤默认 A_raw_window，不额外去漂移；B 去漂移另做对照。

模型预测只乘回一次 d12 scale 得到 raw μV，不添加 target baseline。正式评估不做 median 减除、滤波、再次缩放或 I 替换。Pearson 内部减均值只是相关系数计算。

PPG 不是当前公共 ECG preprocessor 已支持的来源，需要独立处理。不得将 raw counts 转成 ECG 的 μV，也不得直接使用 ECG 的尺度参数。

## 5. 第一版主网络：B5-U-Meta

### 5.1 张量接口

| 输入/输出 | 形状 | 说明 |
|---|---|---|
| x_t | [batch,11,5000] | FM 中间状态 |
| t | [batch] | 连续时间，范围 0–1 |
| anchor I | [batch,1,5000] | raw-scaled，同步可见 |
| 人口学数值 | [batch,3] | age、height、weight |
| 性别 | [batch] | male/female/unknown 类别 |
| 人口学有效 mask | [batch,4] | 四字段真实可用性 |
| v_theta | [batch,11,5000] | 缺失导联速度 |
| 辅助 I 预测 | [batch,1,5000] | 兼容 main 完整 d12 重建监督，见下文 |

导联输出顺序固定为 II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6。

main 的公共 loss 接口要求完整 12 导联可学习输出，并不允许训练中用真实 I 替换预测 I。为兼容该约定，本方案保留一个从 anchor 编码特征预测 I 的小型辅助重建头；I 不进入 FM 噪声状态，也不参与缺失导联评分。训练构造“辅助 I 预测＋11 导联终点估计”，不能把真实 I 拼进去冒充网络输出。

验证默认提供“辅助 I 预测＋ODE 生成 missing11”的完整数组，evaluator 只评分 II–V6，且不替换 I。未来如正式提交接口明确要求保留可见 I，仅由提交适配器处理并注明，不用于制造指标提升。

### 5.2 速度网络结构

~~~text
x_t [11,5000]
  → Conv1d(11→64,k=7)
  → 64通道残差块 ×2                       长度5000，skip0
  → 下采样 stride=2，64→128
  → 128通道残差块 ×2                      长度2500，skip1
  → 下采样 stride=2，128→256
  → 256通道瓶颈残差块                     长度1250
      dilation=1、2、4、8
  → 上采样到2500＋skip1，恢复128通道
  → 128通道残差块 ×2
  → 上采样到5000＋skip0，恢复64通道
  → 64通道残差块 ×2
  → Conv1d(64→11,k=1)
  → 当前速度 v_theta [11,5000]
~~~

残差块可使用 GroupNorm、SiLU、Conv1d，并注入连续时间和人口学条件；保留残差路径与高分辨率 skip。不要在输入层对 anchor/target 另行按窗口标准化或中心化。所有 ODE 步共用同一套网络权重。

上述通道和层数是可执行的起始设计，不是已验证的最优结构。公开预训练、华为基础微调和对应条件消融使用相同骨干。

### 5.3 连续时间编码

t → 128维 sinusoidal embedding → MLP → 256维时间表示；在各残差块投影到该块通道数后注入。t 是 FM 过程时间，不是受试者年龄，也不是物理 ECG 采样时间。

## 6. 条件融合怎么选：按时间关系与表示形态决定

**编码器与融合不是二选一。所有条件都要编码；“一开始使用”或“后续新增”只描述启用阶段。**

| 条件 | 与目标的关系 | 编码方式 | 默认融合方式 | 选择理由 |
|---|---|---|---|---|
| 同步 I | 同记录、同采样时间位置 | 多尺度 1D CNN | 各尺度投影相加 | 保留逐时间位置，直接约束当前形态 |
| 年龄/性别/身高/体重 | 全记录的人口学信息 | 数值＋mask＋类别 embedding → MLP | 各残差块 FiLM | 无逐点时间轴，适合调制通道 |
| 历史 ECG | 同一人、不同采集时刻 | 设备浅层 stem＋masked CNN pooling | 门控 FiLM | 提供个体摘要，避免伪同步 |
| 历史 PPG | 可与历史手表同组，但不同于目标时刻 | 独立 masked CNN pooling | 独立门控 FiLM | 单位/采样率不同；控制噪声和缺失影响 |

这些选择是结合数据关系的工程假设，不是 FM 的强制规定。固定默认实现后，在华为验证集上做有限消融，不能预先宣称 FiLM 或门控一定最好。

### 6.1 同步 I：保留时间位置的多尺度注入

anchor 编码器产生：

- A0：[batch,64,5000]；
- A1：[batch,128,2500]；
- A2：[batch,256,1250]。

对相应速度网络特征 h_s，使用：

\[
h_s'=h_s+P_s(A_s),
\]

P_s 为 1×1 通道投影。可在编码、瓶颈及相应解码尺度使用投影，具体位置写入配置。由于真正同步，这种逐时间位置融合有依据。

第一版不同时叠加 cross-attention。后续若比较拼接融合，使用 concatenate→1×1 projection 保持输出通道和规模接近，只改这一因素。

### 6.2 人口学：MLP＋FiLM

固定连续数值编码，例如 age/100、height_cm/200、weight_kg/150；保留字段 mask。性别统一 male/female/unknown，并使用小类别 embedding。真实缺失的数值可用零占位，但 mask 和 unknown 表示必须保留，不能将其解释成实际 0 岁或 0 kg。

将数值、性别 embedding 和字段 mask 拼接，经小 MLP 输出 z_meta∈R^128。每个残差块生成该块通道数的 gamma、beta：

\[
\tilde h=(1+\gamma_{\mathrm{meta}}(z_{\mathrm{meta}}))\odot h+
\beta_{\mathrm{meta}}(z_{\mathrm{meta}}).
\]

gamma、beta 沿时间维广播，即同一条记录采用同一组通道调制。FiLM 最后一层零初始化，起始调制为恒等映射，之后与主干共同训练。FiLM 是通用特征仿射调制方法；其在 ECG 上的收益需要本任务验证。

人口学编码器从公开预训练开始存在并学习。约 20% 样本丢弃整组人口学条件，并单独模拟字段缺失；这个概率是起始超参数，同步 I 不丢弃。

### 6.3 历史 ECG、PPG：独立编码后门控 FiLM

历史 ECG 先按 main 变换；输入包括有效导联/时间信息。建议 32/64/128 通道的小 CNN，通过 mask-aware 下采样及 masked pooling 输出 128维摘要。不同 ECG 设备使用相应浅层 stem 或设备 embedding，先按单一设备来源独立比较。

历史 PPG 在独立 100 Hz 网格上编码；十秒有效片段约 1000 点，通道数以实际导出为准。建议从输入记录减逐通道记录 median，再使用仅由华为训练 PPG 拟合、冻结的 P95−P5 稳健尺度归一化；零范围与异常值规则单独记录。该 PPG 方案是新增建议，不能声称 main 已实现。保留缺口位置及 time mask，不用 ECG scale，也不强行上采样到 500 Hz。

对历史模态 j∈{ECG,PPG}，定义：

\[
q_j(z_j,h)=\gamma_j(z_j)\odot h+\beta_j(z_j),
\]

\[
h'=\tilde h+m_{\mathrm{ECG}}g_{\mathrm{ECG}}q_{\mathrm{ECG}}+
m_{\mathrm{PPG}}g_{\mathrm{PPG}}q_{\mathrm{PPG}}.
\]

m_j 为该样本的模态可用 mask，g_j∈(0,1) 为 sigmoid 门控，可参考已有配置从 0.05 初始化。gamma/beta 的输出层零初始化，使新增分支起始不扰动已有模型。**不把门控和残差输出同时硬设为恒零**，否则可能阻断两侧梯度。使用上述小非零门控＋零输出层组合，并检查训练后分支能收到梯度。

这一实现是“带门控的 FiLM”，不等于现有配置中 C3 的“FiLM＋预测残差”。若比较 C1/C2/C3，明确各算子、注入位置和参数预算，不能只改名称。

历史条件先注入瓶颈及两个解码尺度；第一轮不与目标逐点相加、不使用 R 峰硬对齐。摘要限制了表达能力，但降低伪同步风险。后续若历史信息确实有用，再考虑少量 context tokens＋cross-attention。

### 6.4 为什么默认不先用 cross-attention

cross-attention 用当前特征作 query、条件 tokens 作 key/value，能学习更细的选择关系，也增加数据需求和每次速度预测的计算量。历史信号没有目标时刻的一一对应，不宜默认用位置相同来约束注意力。

第一版选择多尺度 I 注入和小型 FiLM；已有真实增益后才比较更复杂融合。相同验证患者上比较准确性、raw RMSE、参数量和耗时，再决定保留哪一种。

## 7. Flow Matching 数学定义与损失

### 7.1 训练状态与监督目标

y 为 main 变换后的缺失 11 导联：

\[
y=Y_{\mathrm{raw},II:V6}/S_{II:V6}.
\]

z∼N(0,I) 与该样本独立，t∼U(0,1)：

\[
x_t=(1-t)z+ty,\qquad u_t=y-z.
\]

用有效目标导联 mask m_l 计算：

\[
L_{\mathrm{FM}}=
\mathbb E\left[
\frac{\sum_{l,s}m_l(v_\theta(x_t,t,c)_{l,s}-u_{t,l,s})^2}
{T\sum_l m_l}
\right].
\]

只对合格导联监督，无有效缺失导联的样本跳过。若保留存在坏目标导联的样本，坏导联的 x_t 用噪声占位，不将其坏目标波形喂给其他通道；对应 FM loss 排除。target_quality_mask 只控制监督，不作为推理需要的条件输入。

本方案是独立 noise-target 耦合的线性路径；不宣称使用 minibatch OT，也不允许为匹配噪声而打乱 y 与 I/人口学的正确关联。

### 7.2 起始训练目标

先为少量数据跑通纯 FM 生成，再固定以下主设置，用于同阶段比较：

\[
\tilde y=x_t+(1-t)v_\theta(x_t,t,c).
\]

\[
\tilde Y=\mathrm{concat}(\hat I_{\mathrm{aux}},\tilde y).
\]

\[
L=L_{\mathrm{FM}}
+0.1L_{\mathrm{Huber}}(\tilde Y,Y)
+0.1L_{\mathrm{PCC}}(\tilde Y,Y)
+0.02L_{\mathrm{I}}(\hat I_{\mathrm{aux}},I).
\]

Huber/PCC 复用 main 的 mask primitive；I 辅助项用于兼容完整 d12 监督。系数是建议起始值，不是 main 原公共 loss 权重的复制，也不是已验证最优；需检查量级与梯度，并在匹配对照之间保持一致。

tilde y 是局部速度构造的一步终点估计，通常不等于完整 ODE 结果。验证必须从噪声真实积分，绝不能用验证 target 构造 x_t 来假装生成。t 接近 1 时辅助项容易，可另做 t≤0.8 的辅助损失消融。

第一轮不同时加胸导联加权、均值 head、频域项和复杂生理损失。后续根据新模型误差，只做单因素实验；肢体软约束如启用，复用 main 的 scale-aware、残差去常数形态约束，不对 raw 各导联用硬公式覆盖。绝对电压误差仍由 raw-target 重建项学习。

## 8. 公开数据与华为数据的使用

### 8.1 固定第一公共数据源

推荐先使用 PTB-XL v1.0.3 的 records500：21,799 条十秒十二导联、18,869 位患者，包含 age/sex/height/weight 字段，但不能假定全部字段完整。遵循患者级 folds 1–8 train、9 validation、10 test。

读取时按 WFDB header 的导联名称、gain、baseline、units 还原物理电压，再统一为 μV 和本地导联顺序。不要把 ADC 数值直接当 μV，也不要机械对所有库乘 1000。PTB-XL 的 AVR/AVL 名称及排列按名称映射。

年龄匿名编码超过常规生理范围时，根据官方规则处理，例如 90+ 桶及 top-coded 标记，不把 300 岁作为连续年龄。数据 ID、目标报告、scp_codes、heart_axis 不作为生成条件。

公共十秒记录是合法完整样本；使用独立 public adapter，不机械套用华为原始记录“少于30秒不训练”的筛选规则，否则会误删公共数据。

### 8.2 两个来源学习同一基础任务

公开阶段：

    同记录 public I＋public demographics → 同记录 public missing11

华为基础微调：

    同记录 Huawei I＋Huawei demographics → 同记录 Huawei missing11

这两阶段都使用同步心电图机 I。公开阶段不是学习“华为手表→华为十二导联”，也不将公共同记录六导联假装成华为跨时间设备。

第一版从华为 train-only 去重数据拟合冻结 d12 scales，并在公开预训练、华为微调及验证间保持同一数值表示。公开信号先还原物理单位，再使用该 scale。公共与本地 target 都不减 median、不做逐窗幅值归一化。

固定 scale 仅是数值缩放，不是设备校准。公开与华为仍存在滤波、噪声、人口学、病理和电压分布差异，需要华为微调适应。公开数据提供基础形态覆盖，华为数据决定最终适配与模型选择。

如果固定尺度下公共训练出现数值问题，先诊断单位和训练动态；更改尺度或引入训练统计应另立统一数据表示实验，并让对应对照一起更新。

### 8.3 条件不全时的训练

缺人口学字段使用字段 mask；缺历史模态使用模态 mask。公开阶段不存在 PPG 时关闭其分支，不使用虚构配对。

mask 表达“条件不可用”，不能替代真实条件训练：

- 所有公共样本无 PPG时，公共阶段不会学会 PPG如何帮助生成。
- 后续用华为真实 PPG配对训练新增编码器及融合模块。
- 有 PPG样本中随机丢弃约20% PPG，训练缺失回退。
- 不合格 PPG可关闭分支，保留合格的同步 I/target；历史 ECG的 joint 训练资格仍遵循 main筛选规则。

无需要求每个公开库都拥有全部模态。优先完整同步十二导联、物理单位可靠、患者划分明确；第二公共库和单独 PPG 编码预训练都留到主线有证据后。

## 9. 主训练路线与明确的实验顺序

### 9.1 主线的三阶段权重

~~~text
随机初始化 B5-U-Meta
    ↓ PTB-XL训练集，公开验证集选预训练checkpoint
自己的公开预训练 B5-U-Meta
    ↓ 华为train-only同步数据，小学习率微调
华为适配的 B5-U-Meta，华为validation选checkpoint
    ↓ 新增真实历史ECG/PPG编码器与门控模块
华为扩展版本，各模态独立验证后再组合
~~~

基础版的同步 I 编码、人口学 MLP、FiLM、速度 U-Net和辅助 I头均从公开阶段开始训练。华为微调加载这些权重并继续优化，不默认冻结整个主干。

新增历史分支可短暂冻结主干1–3个epoch做稳定性热身，随后解冻联合小学习率训练。热身是可选工程设置；按日志检查分支梯度，不能永远只训练门控而不适配主干。

### 9.2 按顺序执行的实验清单

| 顺序/ID | 训练路线 | 条件 | 网络 | 必须回答的问题 |
|---|---|---|---|---|
| 0：数据与采样检查 | 少量华为train样本 | I＋人口学 | B5-U | 单位、mask、终点生成、输出还原是否正常 |
| 1：E1 local-meta | 随机初始化→华为 | I＋人口学 | B5-U | 目标域从零训练基准 |
| 2：E2 public-meta | 随机初始化→公开→华为 | 相同I＋人口学schema | B5-U | 公开预训练是否改善华为表现 |
| 3a：E3 public-I | 随机初始化→公开→华为 | 仅I | 相同B5-U主干 | 人口学是否有增量 |
| 3b：E3-local-I | 随机初始化→华为 | 仅I | 相同B5-U主干 | 与华为-only、I-only diffusion比较FM框架 |
| 4a：E4 ECG | 从E2初始化→华为joint | I＋人口学＋历史ECG | B5-U＋ECG分支 | 历史ECG增量 |
| 4b：E5 PPG | 从E2初始化→华为配对 | I＋人口学＋历史PPG | B5-U＋PPG分支 | 历史PPG增量 |
| 5：E6 ECG＋PPG | 从同一E2初始化→华为配对 | 两种历史条件 | B5-U＋两分支 | 组合是否优于各单模态 |
| 6：E7 CT-meta | 随机初始化→公开→华为 | I＋人口学 | CNN＋Transformer速度网络 | 固定FM后骨干是否改善 |
| 7：融合/采样消融 | 对应匹配路线 | 固定 | 固定 | FiLM/门控、步数、样本均值的收益 |

4a与4b均从同一个基础checkpoint初始化，作为独立实验，不强制先把ECG所有优化完成后才研究PPG。E6也从相同基础checkpoint开始，避免未经控制的串行继承造成偏差。

历史分支比较时，基础版E2应在相同pair/患者/窗口子集重新评估；单模态及组合使用相同可比较数据。没有合格PPG时回退基础路径，并另外报告有效PPG子集结果，避免因筛选掉困难样本而形成虚假增益。

如果E2不能稳定优于E1，先检查目标域适配和训练动态，不急着叠加历史模态或换网络。后续可单独试华为:公开=3:1的batch回放，但不是第一轮默认路线。

### 9.3 与B4及既有回归模型的比较

最小框架比较使用E3-local-I与华为-only、I-only diffusion比较；骨干、数据来源、预算、辅助监督、验证样本和实际采样设置须尽量匹配。如果B4也完成公开I-only预训练，则可与E3 public-I比较。不能用“额外公开预训练＋人口学B5”直接证明FM单独优于未预训练I-only B4。

若资源允许，为diffusion补做相同公开预训练＋人口学条件；既有CNN/Transformer回归模型同样需要按当前main的raw-target重新确认实际代码和权重。本文不使用历史汇报指标，也不预先指定哪种模型最终最好。

## 10. CNN＋Transformer版本怎么做

第一版主线采用B5-U。E7是独立骨干实验，FM路径、数据条件、损失和评估保持不变。

建议在U-Net的1250点瓶颈中增加一个粗时间分支：

    1250点、256通道瓶颈
      → stride=5的时间压缩，250个token
      → 2层双向时间Transformer，d_model=256，8 heads
      → 投影并上采样回1250点
      → 与CNN瓶颈残差融合
      → 原有CNN解码与skip
      → 输出11导联速度

该结构增加全局时间关系，也增加每次速度预测成本。它需要从随机初始化完整执行对应公开预训练与华为微调，不把U-Net checkpoint未经核对直接当成同架构初始化；若研究部分权重迁移，另报该策略。

先研究时间Transformer，导联注意力、大型DiT和复杂context cross-attention暂不并入第一版。每项骨干改变应报告参数量、NFE和同硬件耗时。

## 11. 有限的融合消融与条件诊断

默认融合一旦确定，就先完整训练；不是在每个batch动态选择不同融合算法。

| 比较因素 | 默认 | 可选单因素对照 |
|---|---|---|
| 同步I融合 | 多尺度投影相加 | 拼接后1×1投影 |
| 人口学融合 | MLP＋FiLM | MLP特征广播拼接后投影 |
| 历史融合 | 门控FiLM | 无门控FiLM；匹配参数量的门控残差 |
| 历史表示 | masked全局摘要 | 少量token＋cross-attention |
| 网络骨干 | CNN U-Net | CNN＋时间Transformer |

正确人口学、患者级打乱人口学和全缺失人口学在同一验证人群上比较。年龄/性别本地覆盖有限，不用性别打乱制造有效性结论。

历史ECG和PPG分别做正确、跨患者打乱、缺失分支比较。打乱保持设备类型和合理missing pattern，同一患者不同窗之间打乱不构成独立条件诊断。固定同一生成噪声和求解器，避免把采样随机性误认为条件作用。

门控较大或attention图漂亮都不能单独证明条件有用；以同验证子集指标及多种子结果为依据。

## 12. 起始超参数与采样

| 项目 | 起始设置 |
|---|---|
| Optimizer | AdamW，weight_decay=0.01 |
| 公开预训练LR | 1e-4 |
| 华为基础微调LR | 2e-5至5e-5 |
| 新增历史模块LR | 约1e-4；主干较低LR，独立参数组 |
| Batch | 8–16，按显存调整；记录有效batch及累积步数 |
| 时间采样 | uniform[0,1] |
| EMA / grad clipping | 0.999 / 1.0 |
| 元信息/历史dropout | 初始约0.2，独立开关；I不dropout |
| 基础随机种子 | 42；确认可行后重复43、44 |
| 默认求解器 | Heun，16个积分步 |
| 默认输出 | K=1，固定噪声种子 |
| checkpoint选择 | 华为validation r_missing11最大 |

训练epoch/总步数需要依据算力与既有baseline实际预算确定，不把此表称为已调优设置。E1/E2匹配华为优化步数、批次采样和评估频率，并单独报告E2额外公共训练成本。患者窗口数量不同，可采用患者均衡采样，但所有对照统一策略。

公开训练以公开validation选择预训练checkpoint；华为阶段以华为validation选最终checkpoint。任何scale、元信息统计、噪声增强规则均不能用华为validation目标拟合。验证集已用于调参，不将其称为完全独立最终测试集。

Heun每步通常2次速度网络前向：16步/K1约32NFE，16步/K4约128NFE。比较8/16/32步和K=1/4样本均值，记录耗时。K个候选不允许用真实target选最好一个；样本均值可能改善平方误差，也可能平滑波形，PCC和有限模型不保证改善。

不默认做classifier-free guidance、动态幅值阈值或裁剪。z=0不是条件均值，不用它替代多样本实验。I和全局条件在一条轨迹内固定，条件编码可缓存，速度网络仍每步调用。

## 13. 评估、输出与提交

默认评价流程严格沿用main：

1. 模型输出还原为raw μV，仅乘一次冻结d12 scale。
2. 按pair_id（无则target_record_id）分组，按窗口起点排序并检查连续性/完整性。
3. 拼接记录，每记录每个II–V6导联计算PCC；跨记录等权，再跨11导联平均。
4. missing11_mean_rmse_uV 每导联汇总全采样点平方误差后开根，再对11导联平均；当前 main 的 Task2 加分 RMSE 另取 V1–V6 六个胸导联平均。RMSE与PCC的记录权重不同。
5. 训练target质量mask不删除正式评分导联。evaluator不中心化、不滤波、不替换I。

汇总主分为0.5*r1+0.5*r2；Task2加分为RMSE≤70μV时10，否则700/RMSE。main将二者直接相加，该数并非已明确的官方百分制映射。

输出表建议含：Task1/Task2 r_missing11、raw RMSE、Task2 bonus、多训练种子mean±std、NFE、耗时、参数量、公共训练成本。另存数据来源、条件schema和缺失率，明确基础版与历史版的验证子集。

现有evaluator只接收完整5000点缓存窗，不代表任意长度测试全覆盖。提交适配器须保留原始长度，对尾窗padding并去padding，按最终接口导出采样率、导联顺序和长度。若用overlap-add，作为独立推理策略统一比较，并处理边界。

输入条件仅来自测试可见信号与可获得元信息。不能读取目标median、目标诊断、隐藏导联，不能用target构造验证x_t。

## 14. main接入与工程交付（主线代码及后续扩展）

基础主线代码已位于 baselines/B5，使用方式见该目录 README。以下列出主线文件与后续扩展接入点，不改写原始Data：

- baselines/B5/model.py：B5-U速度主干、I编码器、时间编码、辅助I头。
- baselines/B5/conditions.py：人口学MLP、FiLM、历史编码器、门控与mask。
- baselines/B5/flow.py：线性状态/速度目标、Heun采样、样本聚合。
- baselines/B5/metadata.py：externalid→subject_id关联、重复与字段冲突审计。
- baselines/B5/public_adapter.py：PTB-XL folds、物理电压和导联映射。
- baselines/B5/ppg_adapter.py：后续扩展计划，首版未实现；需处理真实配对、时间缺口、独立尺度、有效片段。
- baselines/B5/losses.py：FM及main兼容的辅助loss wrapper。
- baselines/B5/train_public.py、train_huawei.py、validate.py、predict.py：基础主线已实现；train_history.py 为后续扩展计划。
- configs/experiments/b5_*.yaml：每实验的固定网络、条件schema、数据、损失、预算和采样配置。

main 公共模块不提供模型或完整训练循环，B5 分支已增加独立实现，并通过 wrapper 关联人口学信息；PPG 尚未接入，仍需真实配对适配器。

checkpoint必须同时保存architecture_id/config hash、condition_schema_version、阶段、数据版本与split摘要、冻结ECG/PPG scale、元信息映射、solver/steps/K/seed约定、训练预算。新增历史版本继承基础主干时显式列出新模块，保证加载旧基础权重前后、历史分支未启用时预测一致。

main的C1/C2/C3描述目前主要面向回归模型的特征或预测融合；本B5默认调制速度网络特征。若把门控残差用于速度输出，应写作v_total=v_base+gate*delta_v，而不是不经解释地复制最终ECG预测残差公式。

## 15. 验证清单与完成判据

- 数据：训练/验证患者隔离；公共fold正确；target与I严格同窗。
- 数值：单位和导联映射正确；anchor/target未减median；输出只还原一次。
- 条件：人口学从预训练开始学习；缺失mask生效；历史模态正确关联。
- 初始化：新分支初始不扰动基础版，训练后确有梯度；不机械加载B4权重。
- FM：实际ODE验证；mask坏导联不泄漏坏target；生成不访问目标信息。
- 对照：E1/E2先回答公共预训练价值，E2/E3回答人口学价值，E4/E5/E6回答历史模态价值，E7回答骨干价值。
- 评分：固定华为验证患者/窗口、采样噪声与checkpoint规则；比较raw μV RMSE。
- 交付：完整长度尾段、确定性种子、可加载checkpoint与复现README。

条件增多、门控开启、网络变复杂不构成有效性证据。若某模块没有稳定增益，可作为研究对照保留，但不强制加入最终提交。

## 16. 依据与来源

当前main优先依据：

- [main说明](C:/Users/Ashley/Desktop/HW/可用——main分支数据预处理与评估说明.md)。
- [预处理代码](C:/Users/Ashley/Desktop/HW/huawei/ecg12gen/preprocessing.py)与[配置](C:/Users/Ashley/Desktop/HW/huawei/configs/preprocessing.yaml)。
- [训练协议](C:/Users/Ashley/Desktop/HW/huawei/configs/training_protocol_v1.yaml)与[条件融合协议](C:/Users/Ashley/Desktop/HW/huawei/configs/context_fusion_protocol.yaml)。
- [公共loss](C:/Users/Ashley/Desktop/HW/huawei/ecg12gen/losses.py)、[loss配置](C:/Users/Ashley/Desktop/HW/huawei/configs/losses.yaml)、[evaluator](C:/Users/Ashley/Desktop/HW/huawei/ecg12gen/evaluate.py)。
- [同步训练索引](C:/Users/Ashley/Desktop/HW/huawei/metadata/d12_strict_pretrain_index.csv)、[患者划分](C:/Users/Ashley/Desktop/HW/huawei/metadata/subject_split.csv)、[用户信息](C:/Users/Ashley/Desktop/HW/Data/userinfobean.csv)。

PPG索引/QC参考现有材料，实际波形接入另行验证：

- [PPG.csv](C:/Users/Ashley/Desktop/HW/Data/PPG.csv)。
- [PPG配对报告](C:/Users/Ashley/Desktop/HW/ecg_project/reports/ppg_pairing_report.md)。
- [PPG QC报告](C:/Users/Ashley/Desktop/HW/ecg_project/reports/ppg_qc_report.md)。

原始技术与公共数据来源：

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)。
- [PTB-XL v1.0.3官方说明](https://physionet.org/content/ptb-xl/1.0.3/)。
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)：说明Transformer也可作为生成模型骨干，不代表已在本ECG任务验证。
