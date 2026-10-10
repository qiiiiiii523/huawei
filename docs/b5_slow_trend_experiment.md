# B5-U 慢走势辅助监督：配对20轮实验

保持baseline/B5、原U-Net和条件、Flow Matching主目标、原始μV评估、Task1 r选择best.pt的规则。不增加best_score.pt。推理接口没有新输入，不提取真实目标慢曲线，不改predict.py/model.py/flow.py或main数据处理代码。

## 1. 实现定义

两组从同一份微调best.pt的EMA权重重新初始化，使用新优化器、相同seed与训练配置，各20轮。新输出目录保留原E1/E2结果。

| 项目 | Control | Slow |
|---|---|---|
| 配置 | b5_slow_control.yaml | b5_slow_trend.yaml |
| 输出 | results/B5/E3_slow_control | results/B5/E3_slow_trend |
| slow_trend权重 | 0 | 0.1 |
| 初始化、网络、条件、数据、LR、采样 | 相同 | 相同 |
| validate_initial | true | true |
| 轮数/验证间隔 | 20/5 | 20/5 |

额外损失只作用于V1–V6（诊断最差的胸导联）；其余导联仍保持全部原监督。两份配置都声明相同scope，只改变辅助权重。0.1是首个探索设置，不承诺能提高分数。

训练沿用已有端点估计 `x_hat = x_t + (1-t)*v_theta`。对预测/目标各自做FP32、501点约1秒移动平均，然后各自去均值，计算带目标质量mask的Huber损失（delta=1，单位为当前冻结scale下的归一化数值）。常数偏移不参与新增辅助项，但原FM/Huber仍监督原始电压。

`L = 原损失 + 0.1 * L_centered_slow_trend`。

数据或target张量不被修改，不在辅助项里用真实目标的均值校正预测。无效导联在池化前清零，避免NaN污染邻近时间点。推理仍只输入同步I、人口学mask、噪声和采样时间；新损失不参与推理。一步端点辅助监督不等于真实ODE生成效果，成功必须看完整原始验证r。

旧配置省略slow字段时默认关闭，旧损失及日志字段保持兼容。精确--resume会拒绝改变有效损失；实验必须用--init-checkpoint新建运行。旧checkpoint保留模型/尺度兼容性，不需要重新公开预训练。

## 2. 本机交付和服务器同步

本机没有启动训练。功能测试仅CPU前向/导数、序列/日志夹具，没有优化器更新或训练循环。修改需要由用户提交并推送，服务器再：

```bash
cd /home/qht/huawei
git pull --ff-only origin baseline/B5
python -m unittest baselines.B5.tests.test_slow_trend baselines.B5.tests.test_diagnostics baselines.B5.tests.test_b5 -v
```

使用已有BioFlow环境，单卡，两组建议顺序运行在同一GPU。服务器data/Data目录大小写按已有配置保留；新配置继承旧路径，别重新拟合scale。

## 3. 配对预检与冻结同一初始化（不训练）

```bash
python -m baselines.B5.check_slow_pair \
  --init-checkpoint results/B5/E2_finetune_meta/best.pt \
  --snapshot results/B5/E3_slow_shared/initializer.pt \
  --report results/B5/E3_slow_shared/preflight.json
```

要求模型/条件、冻结尺度、当前训练/验证清单与原华为微调checkpoint一致；两组除slow权重和实验名/输出目录外不得改变设置。源文件SHA256用于确认初始化一致。

预检不进行模型前向或训练；快照不覆盖已有不同内容。已有报告时改report文件名；相同内容快照可复用。若数据清单或source checkpoint变更，应新建实验，不跳过核对。

若需降低显存，只修改共享的b5_slow_pair_base.yaml（如batch4、梯度累积2、validation batch4），两组同时生效，然后重新预检。不要只改一组。

## 4. 两组训练：在服务器明确执行

以下才会训练。放在终端/tmux中，保持会话；本机不执行。

对照组：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_slow_control.yaml \
  --init-checkpoint results/B5/E3_slow_shared/initializer.pt \
  --device cuda \
  --execute-training
```

实验组：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_slow_trend.yaml \
  --init-checkpoint results/B5/E3_slow_shared/initializer.pt \
  --device cuda \
  --execute-training
```

这是各20个新epoch，不是原100轮之后沿用优化器的续训。两组初始化完全相同、优化器均新建。

新配置会在第0轮先做真实ODE验证，保存initial_validation.json和可选中的初始化best.pt/last.pt，然后训练1–20轮。第0轮代表本次追加训练尚未更新，不代表模型未预训练。若以后没有超过初始化r，best.pt会保持初始化模型，避免把训练退步的结果当最佳模型。last.pt仍保存真实训练进度。

中断后分别续训，只用对应运行的last.pt：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_slow_control.yaml \
  --resume results/B5/E3_slow_control/last.pt \
  --device cuda --execute-training

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_slow_trend.yaml \
  --resume results/B5/E3_slow_trend/last.pt \
  --device cuda --execute-training
```

不要把对照last.pt以--resume传入实验组。损失变化应使用新输出和--init-checkpoint。

## 5. 评估两组best.pt（目标不参与生成）

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_slow_control.yaml \
  --checkpoint results/B5/E3_slow_control/best.pt \
  --device cuda \
  --output-dir results/B5/E3_slow_control/best_evaluation

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_slow_trend.yaml \
  --checkpoint results/B5/E3_slow_trend/best.pt \
  --device cuda \
  --output-dir results/B5/E3_slow_trend/best_evaluation
```

输出必须新建或为空。原始r、RMSE和综合分仍由现有main evaluator计算。没有额外scale、滤波或真实目标基线校正。

## 6. 全记录快慢和跨窗口连续性诊断

```bash
python -m baselines.B5.diagnose_evaluation \
  --evaluation Control=results/B5/E3_slow_control/best_evaluation \
  --evaluation Slow=results/B5/E3_slow_trend/best_evaluation \
  --slow-seconds 1 \
  --output-dir results/B5/E3_slow_pair_diagnostics
```

仍只读已有数组，在CPU运行。现在额外输出：

- window_mean_diagnostics.csv：每个10秒窗口的预测/目标均值和均值误差。
- window_mean_change_error_rmse_uV：相邻窗口均值误差的变化，检查窗口间漂移不一致。
- point_jump_error_mae_uV：预测边界跳变扣除目标自然跳变后的误差，避免把真实心搏波峰当成拼接问题。
- boundary_100ms_each_side_error_jump_mae_uV：边界前后各100ms平均误差的变化。

边界诊断不参与训练，不拼接/校正预测；没有边界时报告undefined。窗口慢损失本身不保证跨窗口连续性，因此必须一起检查。

## 7. 核对配对条件并比较

```bash
python -m baselines.B5.compare_slow_pair \
  --diagnostics-dir results/B5/E3_slow_pair_diagnostics \
  --output-dir results/B5/E3_slow_pair_comparison

cat results/B5/E3_slow_pair_comparison/report.md
```

核对初始化内容SHA、尺度、清单、训练和采样设置、初始ODE验证一致性，再按原Task1 r选择包含第0轮在内的最佳结果。诊断路径与最佳模型原始r也须匹配。结果含原始r增量、RMSE变化及快慢诊断r。

先看两组是否都完成20轮，再看实验相对对照的原始r是否提高、快变化是否明显退步、RMSE和边界跳变是否恶化。不以辅助训练loss下降判断成功；一次seed的正结果不是稳定收益证明。

打包小报告，不上传完整模型/预测：

```bash
python - <<'PY'
from pathlib import Path
import zipfile
roots = [Path('results/B5/E3_slow_control'), Path('results/B5/E3_slow_trend'),
         Path('results/B5/E3_slow_shared'), Path('results/B5/E3_slow_pair_diagnostics'),
         Path('results/B5/E3_slow_pair_comparison')]
out = Path('B5_slow_pair_reports.zip')
with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv', '.md', '.log'}:
                z.write(p, p.as_posix())
print(out.resolve())
PY
```

## 8. 本地验证范围

测试覆盖：中心慢曲线与NumPy定义一致、常数偏移不影响辅助项、mask/NaN/梯度、旧损失及RNG兼容、无目标采样接口不变、旧checkpoint精确续训兼容和变更损失拒绝、配对配置与快照不可覆盖、初始验证checkpoint保存、变差时选择第0轮、跨窗口跳变与自然目标变化区分。

本机未进行训练或GPU验证；服务器需运行上述功能检查和真正的实验验证。0.1辅助权重是验证中的方案，不宣称已有分数提升。
