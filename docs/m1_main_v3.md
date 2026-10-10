# M1 迁移到当前 main：协议与服务器用法

迁移日期：2026-10-10。分支 `baseline/M1-main-v3`；基于 main `a3613b7`，移植旧分支 `ablation/M1-attention-v2` 的 `e53e900`。保留旧分支及旧结果。本次本机不启动训练。

## 改动范围

保留多尺度 CNN、时间/导联双轴 Transformer、按导联 FPN 解码器，以及 both/time_only/lead_only 注意力实验配置。新增的是 main-v3 适配和缺失 context 可用性控制，没有改变骨干通道、块数或新增可训练参数。

- 读取当前 `task1_output_v2`、`task2_output` 和去重 strict train 索引。
- 同步 I 和 d12 target 保留原始电压，只除冻结尺度；不减 median、不裁剪。
- 目标尺度默认由 strict train 全部去重窗口拟合，与任务选择无关；可用 `--scales` 精确复用 B5 的 NPZ 尺度。重新拟合可能有 NumPy 版本的微小浮点差异，因此比较时直接复用 artifact。
- P1 历史 ECG 使用 main 提供的完整记录 baseline。保留完整验证窗口；context 有任何时间缺口或输入质量不合格时关闭整窗 context 注入。这是原编码器缺乏 masked pooling 时的保守策略，不是已实现的局部缺口恢复。
- 按 pair/record 和 start_sample 拼接后计算缺失 11 导联相关性；Task2 加分 RMSE 只取 V1–V6。验证不替换预测 I，不按训练 QC 删除评分导联。
- 旧目标中心化 checkpoint 缺少 `M1-main-v3` 协议标记，明确拒绝加载。不能仅给旧 checkpoint 添标记假装兼容，应重新训练。
- 正式验证不能限制 batch 数量，避免拼接不完整记录。训练入口必须显式传 `--execute-training`；help、prepare、validate 不训练。
- 输出 `m1_best.pt`、`m1_last.pt`、逐轮 history 和每次验证预测/目标/元信息。当前没有实现 exact resume，不能把 `m1_last.pt` 当成 B5 的续训命令使用。

## 服务器目录

```text
/home/qht/
  task1_output_v2/
  task2_output/
  huawei/                       M1 分支代码
    results/B5/shared/preprocessing_scales.npz
```

M1-P0 不读取人口学和 PTB-XL。B5 的 shared scales 是外部 artifact，不在 Git 中；换目录运行时传它的绝对路径。不要在服务器 B5 运行进程使用的同一个目录切换分支；等进程结束，或使用独立 checkout。

## 先做无训练检查

在已配置的服务器环境中：

```bash
cd /home/qht/huawei
python -m unittest discover -s tests -p 'test_m1*.py' -v
python scripts/prepare_m1.py \
  --scales results/B5/shared/preprocessing_scales.npz \
  --report results/M1_main_v3/preflight.json
```

要求 `ready: true`。prepare 会检查 P0 的两任务验证数据和患者隔离，不训练。没有共享尺度时可省略 `--scales` 并用 `--save-scales results/M1_main_v3/shared/scales.npz` 从 strict train 拟合；已有文件不会被覆盖。

## M1-P0：同步 I-only 回归参照

以下是服务器后续训练指令，本次没有执行。只需训练一个 P0；它的训练集是同一份 strict train，默认由 Task1 验证选模型，然后同一个 checkpoint 评估两个任务。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_m1_axial.py \
  --task-id task1 --stage P0_anchor_only --fusion-mode none \
  --scales results/B5/shared/preprocessing_scales.npz \
  --epochs 150 --validate-every 10 --batch-size 4 \
  --device cuda --output-dir results/M1_main_v3/P0 \
  --execute-training

CUDA_VISIBLE_DEVICES=0 python scripts/validate_m1.py \
  --checkpoint results/M1_main_v3/P0/m1_best.pt --task-id task1 \
  --device cuda --output-dir results/M1_main_v3/P0/evaluation/task1

CUDA_VISIBLE_DEVICES=0 python scripts/validate_m1.py \
  --checkpoint results/M1_main_v3/P0/m1_best.pt --task-id task2 \
  --device cuda --output-dir results/M1_main_v3/P0/evaluation/task2
```

训练保存自己的 `preprocessing_scales.npz`，validate 默认读取 checkpoint 同目录的 artifact 并核对目标尺度。输出目录须为空。

M1-P0 只有 I 条件，B5-meta 还有人口学条件；二者是整体效果参考。要研究纯框架差异，另比较 B5 I-only，并记录参数量、训练预算、采样调用次数及耗时。默认 M1 学习率/损失保留原模型设置，不宣称与 B5 严格匹配。

## M1-P1：保留历史 ECG 实验能力

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_m1_axial.py \
  --task-id task1 --stage P1_joint_anchor --fusion-mode film_gated_residual \
  --context-source-type watch_ecg \
  --p0-checkpoint results/M1_main_v3/P0/m1_best.pt \
  --scales results/M1_main_v3/P0/preprocessing_scales.npz \
  --epochs 100 --validate-every 10 --device cuda \
  --output-dir results/M1_main_v3/P1_watch --execute-training
```

Task2 的 machine/body context 分开训练，分别设置 `--task-id task2 --context-source-type ecg_machine_d6` 或 `body_scale_d6`。对应 P1 验证只包含该设备的完整 pairs，是子集指标；不能冒称完整 Task2，也不能直接与 B5 的全 Task2 分数比较。

单独评估 P1 同样使用 `validate_m1.py`，它从 checkpoint 读取 stage 和 source。注意力消融继续通过 `--m1-config configs/m1_time_only.yaml` 或 `configs/m1_lead_only.yaml` 选择；不同结构 checkpoint 严格校验，不宽松加载骨干。

## 无目标推理

`predict_m1_axial.py` 接收显式 `[N,1,5000]` anchor，不接收隐藏 target。P1 还须提供完整可见 context 记录的 baseline：`--context-baseline-npy`，形状 `[N,C]`，或同一物理记录的 `[C]`。不能从单窗重新估计，也不能跨不同记录共享一个 baseline。

可选 `--context-valid-mask-npy` 表示缺口。默认 `prediction_raw.npy` 和 `prediction_submit.npy` 都保留学习到的 I；仅显式 `--copy-observed-i` 在提交副本替换可见 I，正式验证始终使用 raw 预测。预测按 `--batch-size` 分批处理。当前接口仍限定 5000 点窗口；B-variant 的独立无目标推理预处理尚未提供，入口明确拒绝，不能静默用 A 数据代替。

## 本机验证记录

使用已有隔离 CPU 依赖，不安装全局环境，不启动训练、不调用 optimizer.step。12 项模型/迁移测试通过，包括原模型前向/梯度契约、raw 电压、context baseline/缺口隔离、完整记录评分、旧 checkpoint 拒绝、尺度检查、当前 main loss 梯度、分批无目标推理和 CLI 显式启动保护。

实际缓存只读检查：P0 train 938 窗；Task1 validation 231 窗/21 pairs，Task2 300 窗/25 pairs；训练/验证患者不重叠。P1 watch 859/231 窗，machine-d6 865/240 窗，body-d6 350/60 窗。目标自评分 r=1 只证明评分接口正确，不是模型成绩。GPU训练、收敛和精度尚未验证。
