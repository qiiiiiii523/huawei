# B5-best-exp-V13：V1/V3局部形态修正实验

## 1．分支与实验目的

本分支从`B5-best`的`d45b240`创建，分支名为`B5-best-exp-V13`。V13表示V1和V3；不用斜杠，避免分支名看起来像多级目录。

本地代码worktree：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-V13`。

服务器代码放在新worktree：`/home/qht/huawei_B5_best_exp_v13`。

原数据、原权重和全部新结果仍放在`/home/qht/huawei`。不在原B5-best或B5-S50目录启动新训练。

待验证的假设：V1/V3存在尖峰偏浅、峰宽及局部时间误差；增强最后一级高分辨率解码能力，是否能提高最终完整记录相关性。不能把该假设理解为两个导联必然同时提高。

## 2．只改一个因素

```text
FM最后一级解码特征：[B,64,5000]
   ├ 原1×1输出层 → 11导联速度场
   └ 1×1降到16通道 → 并行Conv1d(5/15/31) → 拼接、SiLU
                                      → 零初始化1×1输出2通道
                                      → 加到V1/V3速度场
```

默认新增14,242个参数。缺失11导联顺序中的下标为5、7；完整12导联顺序中为6、8。核长是实验起始设置，不是已验证最优参数。

FM路径、原FM/Huber/PCC/I辅助损失、同步I、人口学条件保持原定义。新分支从已有解码特征继承条件和生成时间信息。没有新增慢条件注入、峰值真值输入或额外形态损失。

|项目|control对照组|v13实验组|
|---|---|---|
|FM初始化|原B5固定EMA权重|同一权重＋零初始化新分支|
|更新参数|整个原FM|整个FM及新分支|
|S50-F|固定同一权重，不训练|相同|
|损失、学习率、batch等|沿用原FM运行的设置|相同|
|新增训练预算|20轮|20轮|
|周期验证|每5轮，最后一轮也验证|相同|
|生成|Heun16/K16/seed42|相同|
|最终组合|FM−完整记录501点均值＋S50-F|相同|
|慢组合系数|固定β=1|相同|
|选模|最终组合(R1+R2)/2|相同|

固定β=1用于分离模块的作用。seed43/β0.75复核是另外的推理实验，本轮不混入。即使新分支只直接改V1/V3，联合微调和多导联生成状态仍可能影响其他导联，因此必须检查全部导联。

## 3．调用的现有权重

FM初始化：

```text
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt
```

固定S50-F：

```text
/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/best_score.pt
```

历史数据指纹说明：

```text
/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared/prepared.json
```

这些路径来自`configs/b5_best.json`，不是B5-S50慢条件注入训练的权重。预检读取它们，将两个推理模型写入本次`shared/components.pt`，不复制ECG缓存、不修改历史文件。

## 4．服务器更新代码与创建worktree

先将本地`B5-best-exp-V13`提交推送到gitee。然后服务器在BioFlow环境执行以下命令；首次创建只执行一次：

```bash
(
set -e
cd /home/qht/huawei

git fetch gitee B5-best-exp-V13

git worktree add --detach \
  /home/qht/huawei_B5_best_exp_v13 \
  gitee/B5-best-exp-V13

cd /home/qht/huawei_B5_best_exp_v13
git log -1 --oneline
)
```

如果这个worktree已经存在，只更新这个worktree，不在原huawei目录切换分支：

```bash
(
set -e
cd /home/qht/huawei_B5_best_exp_v13
git fetch gitee B5-best-exp-V13
git switch --detach FETCH_HEAD
git log -1 --oneline
)
```

不要在实验运行中更新代码。预检、训练、续训、评估核对同一份实现指纹，防止中途换实现。

## 5．先预检：不训练、不用GPU

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_v13

run=/home/qht/huawei/results/B5-best-exp-V13/local_shape_v1
mkdir -p "$run"

python -u -m baselines.B5_v13.prepare \
  --repository-root /home/qht/huawei \
  --output "$run/shared" \
  --epochs 20 \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

预检应显示`ready:true`、`optimizer_updates:0`、`velocity_max_abs_difference:0.0`，并列出原学习率、batch、20轮预算和Heun16/K16。它检查原权重指纹、训练和验证数据/QC/人口学指纹、患者划分，以及5000点窗口在t=0/0.5/1时的初始速度场一致。

`shared`成功后不用重复prepare。重复运行会因非空目录而退出，这是保护已配对设置。若要改变预算/学习率，用新的运行根目录重新prepare，两组都用新study。

**日志不放进要求为空的输出目录。** 预检日志在`run/prepare.log`；训练日志在`run/control.train.log`、`run/v13.train.log`。不要改成`shared/run.log`或提前创建control/v13里的日志。

## 6．分别用GPU0、GPU2并行训练，并自动评估best与last

下面两条放到两个服务器终端执行。只有CPU预检完成后才能启动。

终端A，GPU0，对照：

```bash
cd /home/qht/huawei_B5_best_exp_v13

bash scripts/run_b5_v13.sh \
  --group control \
  --study /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/shared/study.json \
  --output /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/control \
  --gpu 0
```

终端B，GPU2，V1/V3模块：

```bash
cd /home/qht/huawei_B5_best_exp_v13

bash scripts/run_b5_v13.sh \
  --group v13 \
  --study /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/shared/study.json \
  --output /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/v13 \
  --gpu 2
```

脚本先运行第0轮初始验证（Heun16/K16，可能需要较长时间），核对当前组合分数；然后新增20轮FM训练，每5轮验证，结束后再分别评估`best.pt`和`last.pt`。相同记录在两个任务中复用相同预测。

`best.pt`按**最终组合的QC平均r**选，不按FM单独r、Task1单独r或RMSE选。第0轮也参与选模；若所有续训都没超过基准，best仍是第0轮，这时应同时看last和验证曲线。`last.pt`保存优化器、EMA、随机状态及数据加载器状态，用于续训。

这里保存两份即可：`best.pt`和`last.pt`。本轮只有相关性目标，不再额外生成best_bonus等旧评分文件。

## 7．中断后续训

例如实验组中断，GPU2续训同一个输出目录：

```bash
cd /home/qht/huawei_B5_best_exp_v13

bash scripts/run_b5_v13.sh \
  --group v13 \
  --study /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/shared/study.json \
  --output /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/v13 \
  --resume /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/v13/last.pt \
  --gpu 2
```

对照组同理改为control/GPU0。不要把旧原B5 checkpoint当作本实验`--resume`；它们只用于shared初始化。续训预算是本次总共20轮，不是再加20轮。

## 8．两组完成后比较，不用GPU

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_v13

run=/home/qht/huawei/results/B5-best-exp-V13/local_shape_v1
report="$run/comparison_$(date +%Y%m%d_%H%M%S)"

python -u -m baselines.B5_v13.compare \
  --run-root "$run" \
  --output "$report" \
  2>&1 | tee "${report}.log"

cat "$report/report.md"
)
```

主要文件：

|文件|看什么|
|`comparison_时间/report.md`、`summary.csv`|两组best/last、所选轮次、两个任务r及平均r|
|`lead_comparison.csv`|全部11导联，Task2机器和体脂秤分别保留，是否有其他导联受损|
|`morphology_summary.csv`|V1/V3峰谷幅度比、峰值时间、半高宽误差和未平移局部r|
|`morphology_record_lead.csv`|逐目标细节，I标差时跳过局部心搏诊断|
|PNG|按对照局部导联完整r低/中/高选择的代表记录，同片段原始波形和120秒慢走势|
|每组`validation_metrics.csv`、`history.jsonl`|全过程，是否只是一轮偶然改善|

局部统计复用既有I锚点诊断，先按每条记录统计，再做记录中位数；不是临床峰值标注，也不是正式评分。不会按真值平移/缩放输出，不会更改QC集合。

判断：优先看实验best是否超过同预算对照best；再看V1/V3和局部形态是否改善、其他导联是否退步。不能只因新增损失下降或峰值变大就认定有效。若有收益，再考虑另一个训练seed复核；本轮不同时叠加慢CNN组合PCC训练。

## 9．打包给分析用

```bash
cd /home/qht/huawei

tar \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -czf B5_V13_reports.tar.gz \
  results/B5-best-exp-V13/local_shape_v1
```

包含study/preflight、日志、训练曲线、best/last完整评分、导联明细、局部形态和图片；不包含大型权重和数组。

## 10．新权重的加载与测试输入

实验checkpoint使用独立协议，不能交给旧`baselines.B5.validate`或旧B5-best打包器，避免忽略新增模块。上述脚本已使用`baselines.B5_v13.validate`。

如需单独重新评估，使用新输出目录：

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m baselines.B5_v13.validate \
  --study /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/shared/study.json \
  --checkpoint /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/v13/best.pt \
  --output /home/qht/huawei/results/B5-best-exp-V13/local_shape_v1/v13/manual_evaluation_v1 \
  --device cuda
```

独立推理入口是`baselines.B5_v13.predict`，参数为`--study`、`--checkpoint`、`--anchor-npy`、可选`--metadata-json`、`--record-key`、`--output`和`--device`。I为μV、500Hz、120秒，输出12导联；I辅助头不参与缺失导联评分。推理只读取模型、study和可见输入，不读取训练/验证目标波形或真实均值。

当前只是功能实现，未在本地训练，也没有声称V13已经提高r。
