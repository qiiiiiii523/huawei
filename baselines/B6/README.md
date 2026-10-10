# B6：M1 风格骨干的条件 Flow Matching

本地分支：`B6`，从 `baseline/B5` 的 `dce0abc` 创建。借鉴 `baseline/M1-main-v3` 的 `18c1dda` 中 `ecg12gen/m1_axial.py`：多尺度 CNN、时间/导联双轴注意力、按导联 FPN 解码。B6 不是 M1 回归模型的别名，也没有载入 M1 权重。

## 设计与实验范围

```text
同步 I ── 多尺度 CNN ── 缓存 5000 / 1250 / 250 点特征 ─┐
用户信息 ── 缺失 mask + MLP ── FiLM ────────────────────┤
连续时间 t ── 时间编码 ─────────────────────────────────┤
11 导联状态 x_t ── 每导联状态 CNN ── 250 点 tokens ─────┤
                                                       ↓
                       12 导联 × 时间的 token 网格
                       4 层时间/导联双轴 Transformer
                                                       ↓
                        缺失 11 导联 FPN 解码器
              ＋ 全分辨率局部状态支路和可学习 raw-state 直连
                                                       ↓
                        v_theta(x_t, t, I, 用户信息)
                                                       ↓
                           ODE 生成缺失 11 导联
```

I token 是可见参考，不会被当作缺失导联加噪；另有独立辅助 I 头以保持当前 full12 输出与损失合同。训练和正式验证都不复制真实 I 进预测。缺失导联网格只有 x_t，不输入隐藏 target。

默认：CNN 64/128/256；d_model=128；4层4头；FFN256；250时间token；网络dropout0.1；用户信息定义、FiLM缺失处理与 B5 一致；总参数3,425,799。

raw-state直连按导联分组、零初始化，允许绝对状态电压不经过归一化路径传递。它仍是学习得到的速度项，没有读取目标，不是电压校正或I替换，也不保证恢复未知导联的真实baseline。

保持 B5 的 independent Gaussian linear path、速度 MSE、Huber/PCC/anchor 辅助权重、Heun16、K=1、原始μV评估、train-only冻结尺度、PTB-XL患者fold。慢走势辅助权重默认0；没有历史手表、体脂秤或PPG条件。复用B5已审计的数据/flow/loss组件，B5模块未修改。

保留10秒窗训练和拼接评估；此次没有加入跨窗口状态或记录级连续性约束。更大感受野能否改善慢走势由实验决定，不能保证解决窗口跳变。也是整体骨干实验，不是仅加单个Transformer模块的消融。

`time_only`/`lead_only`配置可做轴向注意力消融。不同架构/条件配置有不同hash，checkpoint不能互换。M1、B5 checkpoint被拒绝；B6必须从头训练，或由B6公开预训练checkpoint微调。

## 单卡与对照设置

默认micro batch=2、gradient_accumulation=4，有效batch=8；验证batch=2。M1风格按导联解码和注意力显存较大，不沿用B5的micro batch8。没有DDP，不使用torchrun。GPU显存/速度未经本机验证。

有效batch与原B5相同，但micro batch分组、网络dropout、参数规模不同；不能声称与旧B5只差一个层。严格骨干对照建议另跑相同micro batch/累积设置的B5控制组，并记录参数数、更新数、训练时长和NFE。与M1回归做框架比较时，优先B6 I-only，M1本身没有人口学条件。

`deterministic: true`时CUDA SDPA使用math实现，避免fused attention backward不确定性；可能增加显存/耗时。改deterministic设置属于新实验，不能作为原运行的等价续训。

## 服务器路径

默认与本机B5配置一致：仓库内 `Data`，两个缓存在父目录，尺度复用 `results/B5/shared/preprocessing_scales.npz`。服务器此前成功配置使用小写`data`时，B6也必须改为相同大小写。

可先在服务器把B5的数据路径复制到B6基础配置，保留B6输出目录：

```bash
cd /home/qht/huawei
python - <<'PY'
from pathlib import Path
import yaml
from baselines.B5.config import load_config
b5 = load_config('configs/experiments/b5_local_meta.yaml')
p = Path('configs/experiments/b6_base.yaml')
b6 = yaml.safe_load(p.read_text())
for key in ('data_root', 'huawei_data_root', 'ptbxl_root', 'demographics', 'scales'):
    b6['paths'][key] = b5['paths'][key]
p.write_text(yaml.safe_dump(b6, sort_keys=False, allow_unicode=True))
print('B6 paths:', b6['paths'])
PY
```

原始数据、缓存和B5尺度不需重复上传。若B5尺度不存在，prepare可显式`--fit-scales`从严格Huawei train拟合；不覆盖已有文件、不从公开/验证目标重新拟合。

## 推荐启动顺序

以下仅为后续服务器命令。本机没有运行优化器更新或训练循环。

1. 提交/推送本地B6后，在服务器fetch并切换B6。B5进程仍在运行时，使用独立checkout或等运行结束，不在其工作目录切换代码。
2. 检查环境/尺度和真实数据；运行CPU功能测试。

```bash
cd /home/qht/huawei
python -m unittest baselines.B6.tests.test_b6 -v
python -m baselines.B6.prepare \
  --config configs/experiments/b6_local_meta.yaml \
  --report results/B6/preflight_huawei.json
python -m baselines.B6.prepare \
  --config configs/experiments/b6_public_meta.yaml --check-public \
  --report results/B6/preflight_public.json
```

要求两份报告ready=true。prepare不训练、不下载。

3. 先做1窗/4窗过拟合，不沿用B5过拟合结果；诊断不会保存供正式训练使用的checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.overfit \
  --config configs/experiments/b6_local_meta.yaml --windows 1 --steps 1000 \
  --device cuda --output-dir results/B6/overfit_w1 --execute-training

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.overfit \
  --config configs/experiments/b6_local_meta.yaml --windows 4 --steps 1000 \
  --device cuda --output-dir results/B6/overfit_w4 --execute-training
```

过拟合关闭网络/人口学dropout，输出settings明确标记；真实固定噪声ODE诊断r应改善、RMSE应下降，再做全量实验。过拟合指标不是官方验证分数。

4. E1：Huawei从头训练。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.train_huawei \
  --config configs/experiments/b6_local_meta.yaml --device cuda --execute-training
```

5. E2：B6公开预训练，再B6华为微调。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.train_public \
  --config configs/experiments/b6_public_meta.yaml --device cuda --execute-training

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.train_huawei \
  --config configs/experiments/b6_finetune_meta.yaml \
  --init-checkpoint results/B6/E2_public_meta/best.pt \
  --device cuda --execute-training
```

不要传B5公开best.pt，不从现有M1回归权重宽松加载。I-only实验使用相应`b6_*_i.yaml`，public-I checkpoint仅能微调I配置。

6. 同一Huawei验证集上评估：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.validate \
  --config configs/experiments/b6_local_meta.yaml \
  --checkpoint results/B6/E1_local_meta/best.pt \
  --device cuda --output-dir results/B6/E1_local_meta/evaluation

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.validate \
  --config configs/experiments/b6_local_meta.yaml \
  --checkpoint results/B6/E2_public_meta/best.pt \
  --device cuda --output-dir results/B6/E2_public_meta/huawei_evaluation

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.validate \
  --config configs/experiments/b6_finetune_meta.yaml \
  --checkpoint results/B6/E2_finetune_meta/best.pt \
  --device cuda --output-dir results/B6/E2_finetune_meta/huawei_evaluation
```

默认EMA；正式r按pair/record拼接缺失11导联，Task2 RMSE加分取V1–V6。CLI打印r和RMSE，目录含overall/lead CSV、raw预测/目标、元信息和competition_score。沿用B5的best.pt按r选模；综合分同时报告，不能把最高r称为最高综合分。当前不额外保留best_score.pt，各epoch历史报告仍可用于多目标分析。

7. exact resume：增加相应配置总epochs后：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B6.train_huawei \
  --config configs/experiments/b6_local_meta.yaml \
  --resume results/B6/E1_local_meta/last.pt --device cuda --execute-training
```

数据、架构、条件、loss、batch/累积、EMA等必须兼容。resume恢复旧optimizer中的LR；新LR或新objective实验用B6 init-checkpoint和新output_dir，不当作完全相同运行。

8. 无目标推理入口：

```bash
python -m baselines.B6.predict --help
```

支持显式可见I、人口学JSON、任意长度输入/尾窗与采样率恢复。仅显式提交选项可复制observed-I；正式验证不复制I。没有任何target输入参数。

## 功能验证与限制

17项B6 CPU测试通过：默认5000点、状态/时间敏感性、注意力轴、梯度、字段缺失、loss质量mask、ODE缓存/分批稳定、I不替换、checkpoint隔离、配置/CLI opt-in、无目标尾窗、记录级评价和无目标CLI。另19项B5共享组件回归/真实缓存只读测试通过。测试有backward以检查梯度，但没有optimizer.step或训练loop。

真实数据prepare通过：Huawei938train窗/78受试者；Task1验证231窗/21pairs，Task2验证300窗/25pairs；PTB-XL17418/2183/2198条，完整records500。数据检查不是模型分数。

CPU依赖为工作区已有隔离库，未修改本机全局环境。没有GPU显存、训练收敛、速度或分数验证；不保证B6优于B5，不保证解决跨窗慢变化。
