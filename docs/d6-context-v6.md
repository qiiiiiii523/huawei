# 心电图机六导联 Context V6

本分支提供 V6 的模型、训练/评估入口、配对配置和已验证权重。B5 源文件保持原样。

## 方法

历史六导联窗口通过多尺度 CNN 编成 token；结合当前 I 导联 query 做双路池化，融合加权均值、标准差和最大值。可靠性头使用 token、有效时间覆盖率和原始窗口权重，生成限制在 ±2 的 logit，调整池化权重。汇总向量通过三层零初始化 FiLM 调节 anchor 特征；每窗 token 供生成网络瓶颈的零门控 cross-attention 使用。可靠性头不直接给 cross-attention 加权。

V6 继承 V5、V4、V3 和 V2 的基础类，所以这些模型模块属于必要运行依赖；旧版本独立实验配置没有全部上传。

## 使用

先按项目原有流程准备本地数据、人口学表、冻结 scales，以及 B5 handoff `results/B5/context_start/best.pt`。这些输入不随本次提交上传。配置路径以项目根目录为准。

训练（将 context 改为 control 即运行配对控制组）：

```powershell
python -m baselines.context_d6_v2.train --config configs/experiments/context_d6_v6_reliability_gate_context_20ep.yaml --device cuda --execute-training
```

评估已保存候选：

```powershell
python -m scripts.eval_context_d6_v6_checkpoint --config configs/experiments/context_d6_v6_reliability_gate_context_20ep.yaml --checkpoint artifacts/v6_reliability_gate/context_seed42/best.pt --output results/context_d6/v6_reliability_gate/recheck_seed42 --device cuda
```

权重 SHA256：`BF43CCBF761FADB5262781B33162F66925C21A38571AE5B8DE995C3189DA847A`，大小 66,064,690 字节。该文件包含训练 checkpoint；评估默认选择其中的 EMA 权重。

## 已有结果与限制

seed42 配对：control r=0.620815255，context r=0.621443631，Δr=+0.000628376；任务二胸导联 RMSE 从 662.377460 降至 661.061601 µV。

seed123、seed2026 的配对 Δr 分别为 +0.000304534 和 +0.000503835；三个种子平均 Δr=+0.000478915。这些均为 validation split（480 窗口、40 条记录）的结果，不是独立测试结论。多轮配置中的训练验证采样/选权重协议仍须结合 resolved_config 核对，不能据此宣称严格战胜所有旧版本。

对应实验记录：`experiments/context_d6/exp010_v6_context_reliability_gate_s42_e20/` 和 `experiments/context_d6/exp011_v6_reliability_gate_multiseed_s123_s2026/`。
