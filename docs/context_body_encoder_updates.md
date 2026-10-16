# 体脂秤 context 编码器更新与复现说明

本次在 `context_body` 分支新增两代**独立记录**条件编码器，不改变原 B5 预测的目标：给定目标时刻的 I 导联，生成缺失的 11 导联。体脂秤六导联只作为同患者、非同步记录的 context，不与目标波形逐点对齐，也不直接复制为目标导联。

## 更新在哪里

| 文件 | 更新内容 |
| --- | --- |
| `baselines/B5/context_body_attention.py` | 在原六导联 CNN 后加入带有效采样点 mask 的局部时间注意力；保留跨窗口加权均值／标准差及原 FiLM 接口。注意力权重零初始化，从 mean+std adapter 继续初始化。 |
| `baselines/B5/context_body_beat_rhythm.py` | 在注意力表示旁增加显式心拍形态 CNN、RR/节律 MLP 和独立小门控。只使用体脂秤 context 检测心拍；每条记录最多选 32 拍，每拍含同步六导联及 mask，RR 用六个稳健统计量。支持 `beat`、`rhythm`、`beat_rhythm` 三种消融。 |
| `scripts/train_b5_body_frozen_adapter.py` | 加入四种 `--encoder-variant`；注意力从 `A_frozen_meanstd_v1/best.pt` 初始化，心拍／节律从 `A_frozen_attention_v1/best.pt` 初始化。验证源 B5 的 SHA、冻结主干和 FiLM，并做梯度预检；各实验使用独立输出目录。 |
| `scripts/evaluate_b5_body_diagnostics.py` | 可加载新 checkpoint，使用同一验证名单执行正确 context、禁用 context 和跨患者打乱对照。 |
| `scripts/check_b5_body_adapter_usage.py` | 支持注意力 checkpoint 的表示／FiLM 使用量诊断。 |
| `scripts/audit_b5_beat_rhythm_similarity.py` | 只读统计新分支和最终 embedding 的跨患者余弦相似度，不保存患者 ID。 |

`docs/context_body_attention_plan.md` 和 `docs/context_body_attention_results.md` 保留第一代注意力试验的协议及详细结果。原有 mean/mean+std 实验代码和结果不覆盖。

## 运行方式

在项目根目录设置 `PYTHONPATH=.`。需有 `torch`、`scipy` 及现有 B5 依赖；训练需 GPU。先准备仓库原有数据清单、体脂秤缓存和预处理参数，以及下列**不随 Git 推送**的权重：

- `results/B5/context_start/best.pt`
- `results/context_body/A_frozen_meanstd_v1/best.pt`（训练注意力时）
- `results/context_body/A_frozen_attention_v1/best.pt`（训练心拍／节律时）

先对所选变体运行 `python scripts/train_b5_body_frozen_adapter.py --encoder-variant beat --preflight`，再去掉 `--preflight` 正式训练；其余变体名称为 `attention`、`rhythm`、`beat_rhythm`。默认协议：A 原始窗口、seed 42、10 epoch、batch 4、梯度累积 2、每 5 epoch 用 EMA/Heun16/K16 验证，并按 `r_missing11` 选最佳。评估示例：

```bash
python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_beat --mode A_true
python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_beat --mode A_disabled
python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_beat --mode A_shuffled
```

`frozen_beat` 可换成 `frozen_rhythm` 或 `frozen_beat_rhythm`。这些脚本默认会将 checkpoint、预测数组和评估报告写入被 Git 忽略的 `results/context_body/`；不要提交比赛原始 ECG、患者清单、预测数组或模型权重。

## 已完成的验证与结论

固定体脂秤验证子集 25 患者配对、300 目标窗口；训练 1,284 窗口。最佳均为第 5 epoch。指标越高越好：

| 模型 | 正确配对 `r_missing11` | 禁用 context | 跨患者打乱 |
| --- | ---: | ---: | ---: |
| 局部注意力 | 0.612042 | 0.612632 | 0.612025 / 0.612015（两个独立打乱种子） |
| 仅心拍 | 0.612021 | 0.612632 | 0.612015 |
| 仅节律 | 0.612019 | 0.612632 | 0.612015 |
| 心拍＋节律 | 0.612000 | 0.612632 | 0.611995 |

新分支与各自跨患者打乱的差值只有约 `0.000005`；三组正确配对的 r 均未超过注意力版本，也低于禁用 context。25 位验证患者均有足够心拍和 RR 候选，但跨患者心拍分支输出的余弦相似度均值约 0.9994，节律分支约 0.9906，最终 embedding 约 0.9756–0.9773；没有观察到明显的患者特异性预测收益。余弦相似度高不等于完全无信息，且此结果只是单随机种子、小验证集的探索性试验，不是比赛最终成绩。

注意力行的两个打乱值来自不同种子；其余三行使用固定跨患者轮换映射，不能跨行把微小差值当成严格配对检验。各权重和完整推理结果仅保留在实验环境的忽略目录中，本次提交只包含实现、复现脚本及聚合说明。
