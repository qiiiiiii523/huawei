# B5-T：仅增加瓶颈时间Transformer残差支路

从精简 `B5` 的 `68f78d0` 创建 `B5-T`。沿用 baselines.B5 的训练/评估/推理入口，配置区分原B5与B5-T；B5分支本身未改。当前只增加窗口内时间建模，不重写CNN、解码器、导联输出、条件、FM路径、损失或预处理。

## 网络

原瓶颈 `[B,256,1250]` → 平均池化为250点 → 1×1投影128维 → 可学习时间位置编码 → 2层时间Transformer（4头，FFN256，dropout0）→ 输出投影回256通道 → 恢复1250点 → 加回原瓶颈 → 原解码器和高分辨率skip。

输出投影的weight/bias全零，加入支路初始行为与原模型一致。旧B5 EMA只加载完全相同的原网络参数；仅允许缺少新增temporal_branch参数，逐项审计，不做任意宽松加载。首次反向传播先给输出投影梯度，投影变成非零后梯度进入注意力和位置编码。所有原参数与新参数都参与微调。

生成时间t、同步I多尺度条件、人口学FiLM仍经过原瓶颈，新增支路处理已含这些条件的特征。时间位置编码表示窗口内位置，与FM进度t不同。窗口仍是500Hz/10秒/5000点，不自动读取相邻窗口或120秒记录；没有历史设备/PPG context。

默认2层，可在新实验配置将temporal_layers改为1。结构改变后必须另开初始化运行，不能用不同层数checkpoint精确resume。B5旧architecture hash与state_dict保持兼容；B5-T有独立B5-T-v1架构ID和hash，加载T权重时自动构造新增支路。

## 服务器：先检查，不训练

本机提交推送本分支后，服务器fetch并切到B5-T。保留服务器自己的数据路径修改和运行中的任务，不强制reset。

```bash
cd /home/qht/huawei
git fetch origin
git switch B5-T
git pull --ff-only origin B5-T

python -m unittest discover -s baselines/B5/tests -v
CUDA_VISIBLE_DEVICES=0 python -m unittest baselines.B5.tests.test_time_transformer.CUDATemporalTests -v

python -m baselines.B5.check_time_init \
  --config configs/experiments/b5_t_finetune_meta.yaml \
  --checkpoint results/B5/E2_finetune_meta/best.pt \
  --report results/B5-T/preflight/zero_init.json
```

首次没有本地B5-T分支时用 `git switch --track origin/B5-T`。zero_init.json应ready=true、速度与辅助I输出的最大差值均为0，并列出新增参数。此检查只做CPU合成整窗前向，不是ECG成绩，不更新参数。已有训练尺度须与checkpoint一致；数据缓存、人口学和尺度文件均沿用B5，无需上传新信号。

CUDA测试包含严格确定性下的注意力及插值反向传播，不调用optimizer.step。为避免原生线性插值CUDA backward的确定性限制，恢复长度使用半像素坐标的index_select/加权求和；确定性模式选择math SDPA。CPU通过与CUDA skipped不能代替服务器GPU通过，确认服务器两项实际ok。

## 后续配对实验（本机不执行）

两组都从同一冻结的原B5微调EMA初始化，原损失、数据、学习率、20轮预算、batch和采样相同，只有temporal_enabled不同。期间验证与最终评估都用Heun16/K16，验证会比旧K1慢。第0轮先保存同一初始化成绩，best.pt仍按Task1 r选；若后续没有超过起点，best仍是第0轮，此时查看history/last的训练结果，不误认为未训练。

先确认原best.pt不会被其他运行覆盖；两个initialization.json应记录相同source_checkpoint_sha256。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_t_control.yaml \
  --init-checkpoint results/B5/E2_finetune_meta/best.pt \
  --device cuda --execute-training

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_t_finetune_meta.yaml \
  --init-checkpoint results/B5/E2_finetune_meta/best.pt \
  --device cuda --execute-training
```

输出分别results/B5-T/control与results/B5-T/temporal，不能覆盖旧B5结果。旧B5仅用于初始化：不要把旧B5的last.pt当作B5-T resume，否则架构/优化器不匹配会拒绝。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_t_control.yaml \
  --checkpoint results/B5-T/control/best.pt \
  --device cuda --output-dir results/B5-T/control/evaluation

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_t_finetune_meta.yaml \
  --checkpoint results/B5-T/temporal/best.pt \
  --device cuda --output-dir results/B5-T/temporal/evaluation
```

predict使用T checkpoint后自动加载时间支路，默认最终采样仍由configs/b5_inference.yaml提供Heun16/K16，不需要单独传模型开关。最终仍比较原始μV完整记录r与RMSE，不能以训练loss或窗口内诊断r声称成功。

本机只做CPU功能/导数检查，没有训练或验证性能提升。服务器显存和运行速度需要实际检查。
