# 当前P最佳权重：三尺度监督配对实验

## 1. 这次做什么

从当前最佳P慢CNN出发，固定同一个control100 FM，只比较是否增加之前ALL20使用的三尺度辅助监督。

| 组别 | 初始化 | 继续训练的损失 | 要回答的问题 |
|---|---|---|---|
| control | 当前P最佳慢CNN | 原P损失：慢标签＋F振荡监督＋最终组合PCC | 原方法继续训练能到多少？ |
| all | 同一份P最佳慢CNN | 原P损失＋0.1×三尺度形状损失 | 三尺度监督能否在新的最佳权重上继续提高r？ |

两组都更新整个慢CNN，网络结构不变。FM不更新，公开阶段不重训，复用原control100缓存。不是从ALL20开始，也不是从公开阶段结束权重开始。

之前三尺度监督曾优于同期继续训练对照，但并非各次实验都超过起点。因此可以迁移到当前P最佳权重测试，不能直接认定它必然继续有效。

## 2. 权重、路径与设置

本地分支：`B5-best-exp-PALL`。

本地工作树：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-PALL`。

服务器新代码工作树：`/home/qht/huawei_B5_p_all`。

读取的当前最佳运行目录：

```text
/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/
├── shared/study.json
└── P/
    ├── best.pt          # 慢CNN训练权重及训练状态
    ├── model.pt         # 完整FM＋慢CNN推理模型包
    ├── model.json
    └── completed.json
```

该P来自慢CNN的公开数据适配＋华为微调；FM保持control100。已提供报告中的起点分数为：

| 权重 | Task1 r | Task2 r | 平均r |
|---|---:|---:|---:|
| P最佳，华为阶段第390次更新 | 0.707346743 | 0.702900846 | **0.705123795** |

代码会核对`best.pt`与`model.pt`的慢CNN状态、FM状态、文件指纹和得分。读取的初始化是上述`P/model.pt`，不会使用历史`shared/components.pt`里的旧慢CNN作为初始化。

| 设置 | 本次两组共同值 |
|---|---|
| 新增训练预算 | **390次更新**，约5轮（156条训练记录、每次2条记录） |
| 验证 | 第0步、每78次更新，以及最后一步 |
| 学习率 | **1e-4** |
| 优化器 | 两组重新建立相同AdamW；复用最佳模型参数，不恢复源训练的优化器动量 |
| 训练seed | **43**；源P最佳训练seed是42，本次两组使用共同的新训练seed |
| 推理 | Heun16、K16、推理seed42、β=0.75，固定FM缓存 |
| 选模 | `R=(Task1 r+Task2 r)/2`，可靠I＋目标导联QC；第0步参与选模 |

同一训练seed在两组映射为相同的框架随机种子和相同记录抽样序列。与已有P训练不是逐步完全相同的续跑；这里需要的是两个新组之间的公平对照。

三尺度定义和旧ALL20一致：在完整120秒、50Hz慢曲线上分解为`慢曲线−MA251`、`MA251−MA751`、`MA751`。这些是近似时间尺度，不是理想频段。每个尺度比较相关性及相对形状，目标标准差下限只从训练集正常导联估计。损失内的整记录去均值不会改变实际输出，不会在测试时使用真实目标。

## 3. 上传代码并在服务器建立工作树

先在本地推送此分支，并按现有方式同步到Gitee。下面服务器命令要求Gitee已经有该分支。

本地PowerShell推送命令（本次仅本地提交，没有代你推送）：

```powershell
git -C "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-PALL" push origin B5-best-exp-PALL
```

服务器在BioFlow环境执行一次：

```bash
conda activate BioFlow
cd /home/qht/huawei
git fetch gitee B5-best-exp-PALL
git worktree add --detach /home/qht/huawei_B5_p_all gitee/B5-best-exp-PALL
```

如果该工作树已经创建，且里面没有本地修改，则更新：

```bash
cd /home/qht/huawei_B5_p_all
git fetch gitee B5-best-exp-PALL
git switch --detach gitee/B5-best-exp-PALL
git log -1 --oneline
```

本次结果全部写入新的`results/B5-P-ALL`目录；原B5、原P训练结果和缓存只读。开始实验后保持这份代码不变，续跑会校验代码及数据指纹。

## 4. 准备一次共同初始化

这一步只检查权重、缓存和QC，并计算训练集的尺度归一化参数，不启动训练。可能需要几分钟读取缓存，GPU没动是正常的。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_p_all

run=/home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1
mkdir -p "$run"

python -u -m baselines.B5_p_all.prepare \
  --source-run /home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4 \
  --updates 390 \
  --validate-every 78 \
  --training-seed 43 \
  --output "$run/shared" \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

看到`ready: true`、`training_started: false`，核对源步骤390、源平均r约0.705124、两组权重相同即可继续。`shared`成功生成后不要再次运行prepare；重复执行会因为非空目录而退出，不需要删除重做。

既有缓存自动从源P的study读取，当前应为：

```text
/home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache/
```

缓存目录名字带PCC20，是因为缓存生成于那次实验；其中FM预测仍可复用，不代表本次初始化用了PCC20慢CNN。

## 5. 两张卡并行：分别打开两个终端

建议在两个tmux会话中执行，避免SSH断开中止训练。下面每条命令会顺序完成该组的训练、best/last评估和完整模型导出。

**终端一：GPU0，control。**

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_p_all

bash scripts/run_b5_p_all.sh \
  --run /home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1 \
  --group control \
  --gpu 0
```

**终端二：GPU1，all。**

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_p_all

bash scripts/run_b5_p_all.sh \
  --run /home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1 \
  --group all \
  --gpu 1
```

两组会先重新评分第0步，复现P起点；若与源分数超出模型包允许的容差，会在训练前停止。完成后各有：

```text
results/B5-P-ALL/from_Pbest_seed43_v1/
├── control.train.log
├── all.train.log
├── shared/
├── control/
│   ├── best.pt                 # 本实验慢CNN训练权重
│   ├── last.pt
│   ├── model.pt                # best对应的完整FM＋慢CNN模型包
│   ├── model.json
│   ├── completed.json
│   ├── validation_metrics.csv
│   ├── latest_best_evaluation.txt
│   └── latest_last_evaluation.txt
└── all/                        # 同样文件
```

如果中断，回到同一组命令末尾加`--resume`即可，例如：

```bash
cd /home/qht/huawei_B5_p_all
bash scripts/run_b5_p_all.sh \
  --run /home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1 \
  --group all \
  --gpu 1 \
  --resume
```

不要把`tee`日志放进原本要求为空的组输出目录；脚本已经处理好日志位置。

## 6. 汇总与判断

等两组脚本都显示Completed后执行：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_p_all
run=/home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1

python -m baselines.B5_p_all.compare --run "$run"
cat "$run/summary.csv"
)
```

汇总包括P起点、control的best/last、all的best/last。主要看：

1. all是否超过**P起点0.705124**；不能仅凭比退步的control好就宣布成功。
2. all是否超过同预算control，Task2是否同步受益。
3. best若仍为第0步，说明观察到的继续训练候选没有超越起点，应保留原P。
4. 若all超过两者，再做另一共同训练seed的配对复核，之后才考虑替换默认部署。

本次周期评分使用已有FM生成缓存＋新慢CNN，和原P报告的评分流程一致。此前提供的报告没有包含P最佳完整FM重新生成的部署验证；不能把缓存评分复现写成已完成端到端部署验证。

如新all确实更好，可以随后在GPU2独立做一次实际FM生成验证；明确传完整`model.pt`，不使用默认旧模型：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_p_all

run=/home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1
out="$run/all/deployment_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=2 python -u -m baselines.B5_best.validate \
  --config /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml \
  --checkpoint "$run/all/model.pt" \
  --device cuda \
  --output-dir "$out" \
  2>&1 | tee "${out}.log"
)
```

## 7. 打包报告给我

训练和两组评估完成后执行，不需要打包权重、缓存和大数组：

```bash
tar -czf /home/qht/huawei/B5_P_ALL_reports.tar.gz \
  --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-P-ALL from_Pbest_seed43_v1
```

下载明确路径：`/home/qht/huawei/B5_P_ALL_reports.tar.gz`。

## 8. 本地已检查什么

CPU合成检查覆盖：control与原P损失完全相同、ALL仅增加旧版三尺度项、坏目标导联梯度为零、坏I整记录不评分、两组初始化及起点分数相同、同seed抽样及状态恢复、模型完整导出与目标无关推理、源文件只读及缓存篡改检测。训练入口必须显式启用执行参数。

测试中禁止优化器step，没有本地训练；这不能替代服务器上的真实数据训练与评估。Python语法、运行脚本及本文Bash命令块已检查，四个尺度函数与历史实现逐段一致。
