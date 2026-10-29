# B8：M1 仅时间注意力回归

B8 从 `B7-C5-Mean-Correction` 的当前数据、QC、冻结 scale 和完整 120 秒 selection 规则出发，但不使用其 FM 速度场、Heun 采样、U-Net 或任何 B7 权重。

网络为远端 `ablation/M1-attention-v2` 中 M1 time-only 候选的回归化实现：同步机器 I `[B,1,5000]` 经 `64/128/256` 多尺度 CNN 产生 250 个时间 token；token 被复制为 **7 个 B7 生成导联** query，并经过 4 个双向 **时间轴**自注意力块。没有 lead attention、没有 cross-record context 拼接、没有 flow state。lead-conditioned F1/F0 skip decoder 直接输出 `[B,7,5000] = II,V1–V6`，与 B7 的输出通道完全一致。I 只作为可见条件输入；其余 5 个肢体导联在 inference 中使用同一 B7 物理公式合成为 d12，供 missing11 评分。

`configs/B8/P.json` 是 public10 anchor-only 预训练，`configs/B8/A.json` 是 Huawei10 适配。两者都保留 B7 的 seed、500 Hz/10 s、frozen train scale、II=5、PCC11、目标导联 QC 与可靠 I 输入筛选。A 阶段虽然从 10 秒 view 训练，验证/选模始终对完整真实 120 秒记录完成 II--V6 macro r，分数为 `0.5*task1 + 0.25*machine + 0.25*body`，即 main 的 task1/task2 1:1 评分。

训练入口：

```powershell
python -m baselines.B8.train --manifest PUBLIC_MANIFEST --config configs/B8/P.json --output runs/B8/P --device cuda
python -m baselines.B8.train --manifest HUAWEI10_MANIFEST --config configs/B8/A.json --init-checkpoint runs/B8/P/best_r.pt --output runs/B8/A --device cuda
```

所有 checkpoint 均以 `B8-M1-time-only-regression-v1` 标识，加载器会拒绝 B7/FM checkpoint，防止误将生成模型权重混入回归实验。
