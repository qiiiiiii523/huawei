# Task 2 心电图机六导联 Context 实验说明

本目录用于在 B5 基准模型上加入“同一患者的历史六导联心电图”条件，只负责
`ecg_machine_d6 / A_raw_window`，不包含体脂秤分支，也不处理 Task 1。

## 1. 要比较的两个实验

- `configs/experiments/context_d6_control.yaml`：控制组。使用心电图机 joint 数据继续训练，但关闭历史 context。
- `configs/experiments/context_d6_context.yaml`：context 组。使用完全相同的目标样本和训练预算，同时输入历史六导联 ECG。

两组都从收到的同一个 B5 `best.pt` 独立初始化。两组应保持 batch、更新次数、学习率、
优化器、EMA、验证采样参数、随机种子和选权重规则一致。

## 2. 输入分别代表什么

- `anchor_i`：目标时刻同步可见的心电图机 I 导联，形状 `[B,1,5000]`。
- `context`：同一患者在其他时间采集的完整六导联历史记录，形状 `[B,W,6,5000]`。
- `target`：目标时刻的十二导联 ECG，只作为训练标签和离线验证答案。
- 人口学条件：年龄、性别、身高和体重，继续沿用原 B5 的编码方式。

历史 context 与 target 不是同一次采集，不能把相同窗口编号或相同采样点当作同步信号。

## 3. Context 如何进入 B5

可以把原 B5 理解为：它先阅读目标时刻的 I 导联，形成三份不同精细程度的摘要，再据此生成十二导联。

新增部分先单独阅读历史六导联：

1. 同一套一维卷积编码器逐个读取历史窗口。
2. 时间 mask 会排除缺口、尾部 padding 和无效采样；设备 QC mask 会排除不可靠导联。
3. 每个窗口得到一份摘要，再按照真实有效采样点数量汇总成一份“整条历史记录摘要”。
4. 这份历史摘要不替代同步 I 导联，而是分别给 B5 的三层同步-I特征提供轻量修正。

每一层修正都可写成：

```text
修正后特征 = 原特征 × (1 + 缩放修正) + 偏移修正
```

“缩放修正”和“偏移修正”由历史记录摘要生成。它们的输出层初始化为全零，因此训练开始前：

```text
缩放修正 = 0
偏移修正 = 0
修正后特征 = 原特征
```

所以刚加载收到的 B5 权重时，新模型与原 B5 的输出完全一致。训练过程中，模型只有在历史 ECG
确实能帮助预测时，才会逐步学习如何调整原 B5 特征。

## 4. Mask 与缺失回退

- 无效采样在进入 context 编码器前归零，并且不参与特征池化。
- 不完整尾窗只使用真实存在的部分。
- 不可靠导联由输入 QC mask 排除。
- 不同长度的历史记录按各自有效采样数加权，不把 padding 当成信息。
- 整条 context 缺失或不可用时，所有修正自动为零，模型退回只依赖同步 I 和人口学条件。

## 5. 为什么控制组仍使用同一个新模型

控制组和 context 组使用相同模型结构，可以避免“一个模型参数多、另一个模型参数少”造成不公平。
控制组会把 context、time mask、window mask 和窗口权重全部置零，因此新增 context 模块不会影响预测。
context 组则允许正确的历史六导联进入同一结构。

## 6. 当前训练和评估设置

当前探索配置使用 Heun4、K=1，以缩短周期验证时间。它只用于早期判断方向。
正式比较时，控制组和 context 组必须同时改为 EMA、Heun16、K=16、seed=42，
并从收到的原始 B5 权重重新开始，不能沿用探索实验权重冒充正式结果。

模型按照机器设备完整记录报告：

- II–V6 的完整记录相关系数；
- V1–V6 的 raw μV RMSE；
- 完整验证预测和对应 metadata。

## 7. 测试与运行

```powershell
$python = "D:\\path\\to\\python.exe"

# 功能测试
& $python -m unittest baselines.context_d6.tests.test_context_d6 -v

# 控制组
& $python -m baselines.context_d6.train `
  --config configs/experiments/context_d6_control.yaml `
  --device cuda --execute-training

# Context 组
& $python -m baselines.context_d6.train `
  --config configs/experiments/context_d6_context.yaml `
  --device cuda --execute-training
```

收到的原 B5 成绩只能作为参考，不能代替“在相同机器设备数据上继续训练”的控制组。
当前阶段仅进行实验开发和比较；最终交付目录、正式结果冻结与提交由负责人确认后再进行。
