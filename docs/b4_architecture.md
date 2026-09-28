# B4：I 导联条件扩散模型

## 1. 方法概述

B4 根据同一时间窗内的心电图机 I 导联生成完整 12 导联 ECG。

- 条件输入：I 导联，形状为 [B, 1, 5000]
- 建模目标：缺失的 11 导联，即 II、III、aVR、aVL、aVF、V1–V6
- 输出：形状为 [B, 12, 5000]
- 数据协议：main 分支定义的 P0 严格同步数据
- 不使用手表、体脂秤、报告或诊断信息

B4 是条件扩散模型。训练时只给缺失的 11 导联加噪，I 导联始终保持完整并作为条件；推理时从 11 导联高斯噪声开始逐步去噪，最后将原始 I 导联原样放回第 0 通道。

## 2. 整体框架

B4 由三部分组成：

1. I 导联多尺度条件编码器；
2. 缺失 11 导联的一维 U-Net 去噪器；
3. DDPM 训练过程与 DDIM 反向采样过程。

    完整 I 导联 -> 多尺度条件编码器 ------------------+
                                                      |
    11 导联带噪信号 -> 条件 1D U-Net -> 预测 v -> 反向扩散
                                                      |
    最终 11 导联 <-------------------------------------+

    输出 = concat(原始 I 导联, 生成的 11 导联)

## 3. 条件 U-Net

### 3.1 输入

    anchor_i       [B,  1, 5000]  完整 I 导联
    noisy_missing  [B, 11, 5000]  当前时间步的缺失导联带噪信号
    timestep       [B]             扩散时间步

### 3.2 I 导联条件编码器

I 导联被编码为三个尺度：

    anchor_i [B,1,5000]
      |
      +-- Conv1d(1->64, kernel=7)
      |     condition_full    [B, 64,5000]
      |
      +-- SiLU + Conv1d(64->128, stride=2)
      |     condition_half    [B,128,2500]
      |
      +-- SiLU + Conv1d(128->256, stride=2)
            condition_quarter [B,256,1250]

条件特征不是只在输入层拼接一次，而是在 U-Net 的三个尺度分别注入，使去噪器在高分辨率波形、局部形态和低分辨率语义层面都能访问完整 I 导联。

### 3.3 缺失导联去噪器

    noisy_missing [B,11,5000]
      |
      +-- Conv1d(11->64) + condition_full
      +-- ResidualBlock(64) ---------------- skip1
      +-- Downsample(64->128) + condition_half
      +-- ResidualBlock(128) --------------- skip2
      +-- Downsample(128->256) + condition_quarter
      +-- ResidualBlock(256, dilation=2)
      +-- ResidualBlock(256, dilation=4)
      +-- Upsample(256->128) + concat(skip2)
      +-- ResidualBlock(256->128)
      +-- Upsample(128->64) + concat(skip1)
      +-- ResidualBlock(128->64)
      +-- Conv1d(64->11)
      |
      model_output [B,11,5000]

主干通道：

    64 -> 128 -> 256 -> 128 -> 64 -> 11

时间长度：

    5000 -> 2500 -> 1250 -> 2500 -> 5000

中间层使用 dilation 2 和 4 的扩张卷积，以扩大时间感受野。

### 3.4 时间步编码

    t
    -> SinusoidalEmbedding(128)
    -> Linear(128->512)
    -> SiLU
    -> Linear(512->128)

时间向量注入每个残差块，使网络知道输入当前处于高噪声阶段还是低噪声阶段。

## 4. 扩散训练

设真实缺失导联为 x0，高斯噪声为 epsilon。前向加噪为：

    xt = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon

B4 使用 200 个训练时间步和 cosine 噪声日程。I 导联不参与加噪。

### 4.1 v-prediction

B4 预测：

    v = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x0

恢复干净波形：

    x0_pred = sqrt(alpha_bar_t) * xt - sqrt(1-alpha_bar_t) * v_pred

v-prediction 避免在最高噪声时间步除以非常小的 sqrt(alpha_bar_t)，比旧版 epsilon-to-x0 计算更稳定。

### 4.2 损失函数

    L = L_v + 0.1 * L_x0 + 0.1 * L_corr

- L_v：预测 v 与真实 v 的均方误差；
- L_x0：恢复波形与真实波形的 Smooth L1；
- L_corr：逐导联时间序列相关系数损失；
- 所有损失只作用于质量掩码允许的缺失导联。

训练还使用 EMA 0.999、梯度裁剪 1.0，并按验证集 r_missing11 保存最佳 checkpoint。

## 5. 推理

推理从随机噪声开始：

    xT ~ N(0,I), shape = [B,11,5000]

随后使用 100 步 DDIM 反向采样：

    xT -> x(t-1) -> ... -> x1 -> x0

默认 eta=0。相同检查点、I 导联、初始噪声和随机种子会得到相同结果；改变初始噪声可以得到不同候选结果。

每一步都执行一次完整 U-Net，因此 B4 推理约需 100 次网络前向。预测的 x0 还会经过动态阈值限制，防止异常振幅在反向扩散中被逐步放大。

最终：

    output[:, 0:1] = 原始 anchor_i
    output[:, 1: ] = 生成的 11 导联

## 6. B4 与 B1 的区别

B1 和 B4 都使用一维 U-Net，但它们不是同一种模型。

| 对比项 | B1 | B4 |
|---|---|---|
| 模型范式 | 直接回归 | 条件扩散生成 |
| 输入 | 1 或 6 导联 | 完整 I、当前 11 导联噪声、时间步 |
| 输出 | 一次输出最终 12 导联 | 每一步预测 11 导联的 v |
| U-Net 通道 | 16/32/64/128 | 64/128/256 |
| 下采样次数 | 3 | 2 |
| 条件注入 | 输入信号本身 | I 导联在三个尺度注入 |
| 时间步编码 | 无 | 有 |
| 训练 | 一次前向后直接与真值计算损失 | 随机选择时间步、加噪并学习去噪 |
| 推理 | 一次前向 | 从噪声开始迭代约 100 次 |
| 输出含义 | 单个确定性预测 | 条件分布中的一个样本 |

B1 学习直接映射：

    Y_pred = f(I)

B4 学习条件分布的反向去噪过程：

    p(Y_missing | I)

“都使用 U-Net”只表示二者使用了相似的编码器—解码器骨干，不表示训练目标和推理方式相同。

## 7. B4 与 CNN + Transformer 模型的区别

项目中的 B2 和 M1 属于直接回归或重建模型：

- B2 将 ECG 切成 patch，使用 Transformer Encoder 后一次性解码 12 导联；
- M1 用 CNN 提取局部多尺度特征，用 Transformer 在时间维和导联维建模长程关系，再一次性解码 12 导联；
- B4 用 CNN U-Net 作为扩散去噪器，在每个扩散时间步重复调用。

| 对比项 | CNN/Transformer 回归 | B4 扩散生成 |
|---|---|---|
| 网络输入 | 已知导联 | 已知 I、带噪 11 导联、时间步 |
| 网络输出 | 最终 ECG | 当前步骤的去噪方向 v |
| 训练目标 | 直接逼近真实 ECG | 学习所有噪声等级下的反向过程 |
| 推理次数 | 通常 1 次 | 通常 100 次 |
| 输出随机性 | 通常无 | 可由初始噪声控制 |
| 多解表达 | 容易趋向条件均值 | 可以表达条件分布并采样 |
| 推理速度 | 快 | 慢 |
| 优化难度 | 相对简单 | 更复杂，对采样稳定性敏感 |

CNN、Transformer 和 U-Net 描述的是网络结构；回归模型和扩散模型描述的是学习与生成方式。这是两个不同层级的概念。

同一个 U-Net 可以这样使用：

    输入 I -> 直接输出 12 导联

此时它是回归模型。也可以这样使用：

    输入 I + xt + t -> 预测当前去噪方向

此时它是扩散模型中的去噪网络。Transformer 同样可以作为扩散模型的去噪器，所以不能仅根据是否使用 CNN、U-Net 或 Transformer 判断模型是不是生成模型。

## 8. 生成模型与回归模型的直观区别

### 8.1 回归模型

回归模型像一次性直接答题：

    看到 I 导联 -> 给出最可能的 12 导联

如果同一个 I 导联可能对应多种合理胸导联，普通 MSE 或 Huber 回归容易给出这些结果的平均值，波形可能因此变平滑。

### 8.2 扩散生成模型

扩散模型像从草稿逐步修改：

    随机噪声
    -> 根据 I 导联修正一点
    -> 再修正一点
    -> ...
    -> 得到完整 11 导联

它学习的是不同噪声等级下应该朝哪个方向去噪，可以从不同初始噪声生成多个与 I 导联一致的候选结果。

### 8.3 为什么网络看起来可以一样

网络本质上只是一个可学习函数。决定模型范式的是：

1. 网络接收什么输入；
2. 网络预测什么目标；
3. 如何构造训练样本；
4. 推理时调用一次还是反复调用；
5. 是否从随机变量开始生成。

所以 U-Net 既可以是回归网络，也可以是扩散模型的去噪网络。二者外形可能相似，但学习的问题不同。

## 9. 仓库各模型的真实输入关系

这里必须区分两个概念：

- anchor：目标时刻、与 12 导联目标严格同窗同步的心电图机 I 导联；
- context：同一受试者的辅助 ECG，可以来自其他时刻，不要求与目标逐采样点对齐。

最新版 joint-anchor 协议规定：

| 模型 | P0 输入 | P1 Task1 额外上下文 | P1 Task2 额外上下文 | 输出方式 |
|---|---|---|---|---|
| B0 | 同窗 machine I | watch I | machine d6 或 body d6 | 线性直接回归 |
| 新 B1 | 同窗 machine I | watch I | machine d6 或 body d6 | U-Net 直接回归 |
| B2 | 同窗 machine I | watch I | machine d6 或 body d6 | Patch Transformer 直接回归 |
| B3 | 同窗 machine I | watch I | machine d6 或 body d6 | CNN + time Transformer + FPN 直接回归 |
| M1 | 同窗 machine I | watch I | machine d6 或 body d6 | CNN + lead/time axial Transformer 直接回归 |
| B4 | 同窗 machine I | 不使用 | 不使用 | 条件扩散生成 |

因此，Task2 出现六导联不表示目标时刻的 anchor 变成六导联。正确关系是：

    目标时刻：
      machine I(C) --------------------+
                                        +--> 预测目标时刻 d12(C)
    同受试者其他记录：
      machine d6(B) 或 body d6(A/B) ---+

P0 只使用上面的 machine I(C)。P1 才允许下面的 d6 context 参与条件融合。

### 9.1 为什么仓库中旧 B1 看起来输入六导联

仓库中存在两套 B1：

1. origin/baseline/B1：旧 official-v1 协议；
2. origin/新B1：新版 joint-anchor 协议。

旧 B1 的代码将 Task1 定义为 1 导联直接输入，将 Task2 定义为 6 导联直接输入。因此旧 B1 的 Task2 学习的是：

    d6 -> d12

新版 B1 的 P0 则明确要求：

    same-window machine I -> same-window machine d12

六导联只在 P1-C3 中作为 context，通过 FiLM 和 gated residual 条件模块影响 anchor backbone。

所以旧 B1 并不一定是代码写错，而是采用了旧实验协议。但如果现在要和 B2、B3、M1、B4 在最新版 main 的 joint-anchor/P0 协议下公平比较，那么旧 B1 Task2 的六导联直输结果不应混入该比较，应使用 origin/新B1。

### 9.2 B2 的输入

B2 最新分支会把同窗 machine I 放进 12 导联规范位置，其余导联填零，同时附加 lead mask，然后切成 patch：

    machine I + lead mask
    -> patch embedding
    -> 3 层 Transformer Encoder
    -> 一次性解码 d12

P1 时，anchor 和 context 分别编码成 token 后再做表示层融合。Task2 的 d6 是 context，不替代 I anchor。

### 9.3 B3 的输入

B3 的 anchor backbone 明确只接受：

    anchor_i [B,1,5000]

P0 路径完全不读取 context。其结构为多尺度 CNN、时间 Transformer 和 FPN 导联解码器，一次前向输出 d12。

P1-C3 才额外读取 watch I 或某一种 d6 context，并通过 FiLM 与 gated residual 融合。machine d6 和 body d6 每次实验互斥，不会拼接在一起。

### 9.4 M1 的输入

M1 的 P0 同样只接受同窗 I 导联。CNN 提取多尺度局部波形，axial Transformer 分别执行时间维注意力和导联维注意力，再直接解码 12 导联。

P1 才加入 watch I、machine d6 或 body d6 context。M1 和 B3 的主要区别是 Transformer 的组织方式：

- B3 主要在时间 token 上做 Transformer；
- M1 构造 12 导联乘 250 时间 token 的网格，分别做时间轴和导联轴注意力。

### 9.5 B4 为什么没有六导联输入

B4 是刻意设计的 I-only baseline：

    条件 = 同窗 machine I
    生成目标 = 同窗缺失 11 导联

B4 暂不实现 P1，也不读取 watch、machine d6、body d6、报告或诊断。Task1 和 Task2 的区别体现在各自验证样本和评价分层，而不是 B4 网络输入通道数；两者的模型条件始终都是 I 导联。

## 10. 一句话总结

    B1/M1/B2：输入已知导联，一次前向直接回归完整 ECG。
    B4：输入 I 导联和随机噪声，用 U-Net 反复去噪生成缺失 11 导联。

B4 的生成性不是来自 U-Net 这个名称，而是来自随机加噪训练、时间步条件、反向扩散以及从噪声开始的迭代推理。
