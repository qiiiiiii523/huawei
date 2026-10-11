# B5-F：两级解码器的三尺度加权融合

本分支从稳定 B5（68f78d0）创建，使用 baselines.B5 的原训练、验证和推理入口。
F 指当前窗口内的多尺度解码融合。没有时间 Transformer，也没有新增历史设备、PPG 或跨窗口 context。

## 网络与初始化

原编码器产生三个尺度，默认输入为500Hz、10秒、5000点：

| 特征 | 形状 | 含义 |
|---|---|---|
| skip0 | B×64×5000 | 高分辨率局部特征 |
| skip1 | B×128×2500 | 中分辨率特征 |
| bottleneck | B×256×1250 | 原卷积瓶颈特征 |

两个解码级各新增一个融合残差。三个特征先以 nearest 对齐到该级长度，再分别用1×1卷积投影为64通道。
每级用独立的3个可学习logit，经softmax得到三个尺度的权重，初始各为1/3。
加权和经过GroupNorm、SiLU和输出投影，加入原merge及同步I条件之后、原解码残差块之前。
半分辨率级输出128通道，全分辨率级输出64通道；各级参数不共享。

```text
skip0 ─┐                         原up＋skip拼接＋merge＋I条件
skip1 ─┼→ 对齐长度 → 三路投影 → 加权和 → norm/SiLU → 零输出投影 ──相加
瓶颈 ──┘                                                        ↓
                                                        原解码残差块
```

原拼接融合和所有原网络权重保留。这是用残差形式增加融合能力的消融，不能解释为完全替换解码器。
同步I、人口学FiLM、生成时间t、FM路径、损失和raw μV预处理沿用原B5。
输出仍是11个缺失导联的速度场，辅助I输出不变。推理无需真实目标或其均值/慢曲线。

两个新增输出投影的weight/bias均为零，加载同一个旧B5 EMA后，起步预测与原B5一致。
新增参数的构造隔离CPU RNG，不改变原参数初始化及后续噪声随机流。
第一次反向传播先学习输出投影；输出投影非零后，梯度进入尺度权重和各尺度投影。这是零残差初始化的预期行为。
原网络及新增分支都参与联合微调。尺度权重是每级全局学习的权重，不是逐时刻注意力；其数值不能单独当作重要性解释。

`fusion_enabled: false` 保持原B5架构、state_dict和hash；开启为独立的 `B5-F-v1` 架构。
旧B5只能通过 `--init-checkpoint` 初始化F。仅允许缺少新增fusion_branch参数，原参数逐项核对；不接受任意部分加载。
`--resume` 必须使用本组同架构的last.pt，保留optimizer/EMA/RNG；不能把旧B5的last.pt当成F的resume。
predict根据checkpoint自动恢复融合模块，最终默认Heun16、K16，无需传模块开关或采样参数。

## 服务器先做检查

本机提交并推送B5-F后，在服务器操作。保留服务器尚未提交的数据路径修改，先处理git提示，不要强制reset。

```bash
cd /home/qht/huawei
git fetch origin
git switch --track origin/B5-F
```

若服务器已建立本地B5-F分支，改为：

```bash
git switch B5-F
git pull --ff-only origin B5-F
```

数据目录沿用 `data/userinfobean.csv`、上一级task1_output_v2和task2_output、已有冻结尺度。
不要重新fit scales；应复用初始化checkpoint对应的 `results/B5/shared/preprocessing_scales.npz`。

```bash
python -m unittest discover -s baselines/B5/tests -v
CUDA_VISIBLE_DEVICES=0 python -m unittest baselines.B5.tests.test_decoder_fusion.CUDAFusionTests -v

python -m baselines.B5.prepare \
  --config configs/experiments/b5_f_finetune_meta.yaml \
  --report results/B5-F/preflight/data.json

mkdir -p results/B5-F/shared
cp -n results/B5/E2_finetune_meta/best.pt results/B5-F/shared/initializer.pt

python -m baselines.B5.check_fusion_init \
  --config configs/experiments/b5_f_finetune_meta.yaml \
  --checkpoint results/B5-F/shared/initializer.pt \
  --report results/B5-F/preflight/zero_init.json
```

预检应ready=true、optimizer_updates=0、velocity_max_abs_difference=0和anchor_max_abs_difference=0。
它只做CPU合成整窗前向与尺度兼容性检查，不是ECG成绩、不更新参数。
CUDA测试只检查严格确定性下的反向传播，不调用optimizer.step；本机CPU通过不能代替服务器GPU通过。
`cp -n` 只在共享初始化尚不存在时复制；已有文件不会覆盖。确认其sha与之前T/C实验initialization.json中的source_checkpoint_sha256一致，才能把历史C/T纳入同一比较。

## 配对训练

| 项目 | C：原B5 | F：新增融合 |
|---|---|---|
| 配置 | b5_f_control.yaml | b5_f_finetune_meta.yaml |
| 初始化 | 同一shared/initializer.pt的EMA | 同一shared/initializer.pt的EMA |
| 训练预算 | 20轮 | 20轮 |
| 学习率、batch、损失、数据 | 相同 | 相同 |
| 期间与最终采样 | Heun16、K16、seed42 | Heun16、K16、seed42 |
| 输出 | results/B5-F/control | results/B5-F/fusion |

每组单卡即可，可顺序使用同一张卡。先执行一组完成后再执行另一组；若并行，分别选择空闲卡。

```bash
set -o pipefail
mkdir -p results/B5-F/control results/B5-F/fusion

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_f_control.yaml \
  --init-checkpoint results/B5-F/shared/initializer.pt \
  --device cuda --execute-training \
  2>&1 | tee results/B5-F/control/train.log

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_f_finetune_meta.yaml \
  --init-checkpoint results/B5-F/shared/initializer.pt \
  --device cuda --execute-training \
  2>&1 | tee results/B5-F/fusion/train.log
```

每5轮验证，训练前额外做第0轮验证。best.pt始终按Task1完整记录r选择，last.pt保留训练后的状态。
若训练没有超过初始化，best仍是第0轮，这不是没训练。应同时看history与第20轮验证结果。
两组initialization.json记录源权重sha、数据manifest、尺度sha和加载审计。只能在这些一致时作配对比较。
原b5_finetune_meta.yaml的训练设置未变；K16和第0轮验证仅在本次F/C实验配置启用。

之前T实验的C可以作为历史对照，但必须核对初始化sha、数据/尺度manifest、完整训练及采样配置均一致。
为了排除代码版本或运行差异，本分支提供重新运行C的命令；已有严格匹配的C结果时无需机械重复训练。

## 最佳模型与训练后模型分别评估

下列目录须为空；重复评估请换新的输出目录。

```bash
for arm in control fusion; do
  if [ "$arm" = control ]; then
    config=configs/experiments/b5_f_control.yaml
  else
    config=configs/experiments/b5_f_finetune_meta.yaml
  fi
  for checkpoint in best last; do
    CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
      --config "$config" \
      --checkpoint "results/B5-F/$arm/$checkpoint.pt" \
      --device cuda \
      --output-dir "results/B5-F/$arm/evaluation_$checkpoint"
  done
done
```

训练epoch_0020报告已评估当轮EMA，用于判断训练后的结果；last.pt独立评估可核对其确切输出。
仍以原始μV、拼接后的完整记录Task1/Task2 r与胸导联RMSE判断。F先与同预算C比较，再与共同初始化比较。
训练loss下降、新尺度权重变化、单个窗口图更好，均不能单独证明正式评分提升。

## 打包分析报告

不上传checkpoint、原始缓存和大的波形npy时，可以这样收集报告：

```bash
find results/B5-F -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \) \
  -print0 > /tmp/b5_f_report_files.list
tar --null -czf B5-F_reports.tar.gz -T /tmp/b5_f_report_files.list
```

本机只实现代码并执行CPU功能/梯度检查，没有训练，不承诺F一定优于原B5。TF留待F的结果出来后决定。
