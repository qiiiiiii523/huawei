# B5三组实验：训练搭档适配与分尺度监督

## 1. 本轮实验是什么

在同一分支B5-best-exp-Control100PCC完成三组实验，不再新建模型分支。
三组从同一PCC20慢CNN初始化，网络、S50输入、人口学、优化器、学习率、种子、20轮预算、每5轮验证和β=0.75相同。

|组|命令中的group|训练快残差来源|损失|比较目的|
|---|---|---|---|---|
|A|control_old_fm|原FM缓存|原慢监督＋F监督＋原组合PCC|继续训练对照|
|B|adapt_control100|control100缓存|与A完全相同|B对A：换训练搭档|
|C|adapt_multiscale|与B完全相同|B的损失＋分尺度辅助项|C对B：分尺度监督|

三组最终都与同一个control100快残差组合，使用可靠I＋目标导联QC、完整120秒、两个任务平均r选best.pt。
这里的A组也有组合PCC，不是旧SlowPCC实验中没有组合PCC的control组。
本轮不更新FM，不重新生成缓存，不改网络、推理方法或部署默认模型。

三组可以并行。C对A同时改变了两个因素，不能单独归因；应分别看B对A、C对B。
这个设计检验“在control100搭档下分尺度监督是否有效”，没有额外检验它在原FM搭档下的效果，因此不是完整2×2交互实验。

## 2. C组增加什么损失

预测慢曲线和真实慢标签都使用完整120秒、50Hz的6000点曲线。真实慢标签仍来自原MA501定义，不更改缓存标签。
对两条曲线做完全相同的拆分，L5、L15分别为251点、751点居中移动平均，端点复制：

```text
较短起伏：s - L5(s)
中等起伏：L5(s) - L15(s)
较长走势：L15(s)
```

这是约5秒以下、5～15秒、15秒以上的变化诊断，不是严格频率分带，也不会改变50秒输入范围。
对每种变化分别比较预测和标签：

```text
该尺度损失 = 1 - PCC + 0.1 × 归一化形状Huber
L多尺度 = 三个尺度损失的平均
C组总损失 = 原损失 + 0.1 × L多尺度
A/B组总损失 = 原损失
```

PCC要求走势对应，形状Huber补充约束相对起伏强度，避免只提高相关性但幅度进一步缩小。
新增Huber不增加均值监督，不改变原损失中的绝对电压监督。
默认辅助系数0.1是首轮实验设置，不是已验证最优值；在prepare时冻结，三组运行中不根据验证结果动态改变。

新增项只在自身计算中，按每条完整120秒曲线、每个尺度去除常数均值，并用该训练标签的尺度标准差归一化。
这不修改模型输出或原始target，也不在测试中使用真实目标均值/标准差。测试仍然仅输入I及人口学，输出组合方式不变。
坏目标导联不参与该损失；接近常数的标签也跳过，防止把微小噪声放大成强监督。
接近常数的阈值只从156条训练目标的正常QC导联统计：每个导联每个尺度取max(训练标准差中位数×0.1, 1e-4归一化单位)。
阈值、训练统计、辅助系数、尺度定义均写入study.json，验证数据不参与这些统计。
训练history.jsonl额外记录三个尺度的PCC/形状Huber、有效比例及辅助项加权贡献。

## 3. 本地路径、提交与推送

```text
分支：B5-best-exp-Control100PCC
本地工作树：C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Control100PCC
```

本机PowerShell：

```powershell
cd "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Control100PCC"
git push origin B5-best-exp-Control100PCC
```

本机origin是GitHub，服务器使用Gitee；先按原有镜像流程同步该分支到Gitee。
本次修改了实验协议与代码指纹，三组须新建study。旧partner_v1若已准备或训练，保留它及对应旧代码，不覆盖、不用新代码强行resume。
旧原始缓存仍直接复用，不需要再生成156条FM预测。

## 4. 服务器创建三组实验工作树

本次使用新的代码目录/home/qht/huawei_B5_control100_pcc_ms，避免影响此前两组版本可能已经创建的study或进程。

```bash
conda activate BioFlow
cd /home/qht/huawei

git fetch gitee B5-best-exp-Control100PCC
git worktree add --detach /home/qht/huawei_B5_control100_pcc_ms FETCH_HEAD

cd /home/qht/huawei_B5_control100_pcc_ms
git log -1 --oneline
```

已经存在这个_ms目录时不要重复add。只在没有运行中的任务时，在该目录fetch并merge --ff-only；已有study不随意更换核心代码。
不切换原huawei或旧SlowPCC训练工作树。

## 5. 一次性CPU预检，不生成FM缓存

直接读取现有两个来源：

```text
A的原FM训练缓存：
/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1/cache

B/C训练与三组验证的control100缓存：
/home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache

三组同一control100＋PCC20初始化：
control100_pcc20_cache_v1/shared/components.pt
```

执行以下整段。日志放在shared目录外，不会造成“输出目录非空”：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_control100_pcc_ms

run=/home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1
mkdir -p "$run"

python -u -m baselines.B5_control_pcc.prepare \
  --old-run /home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1 \
  --control100-run /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1 \
  --reference-report /home/qht/huawei/results/B5-rescore-reliable-I/default_20261017_192455/correlation_score.json \
  --epochs 20 \
  --multiscale-weight 0.1 \
  --output "$run/shared" \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

这是CPU缓存校验、训练标签尺度统计和一次慢CNN前向检查，显卡不动正常。
程序检查来源SHA、输入I/人口学/目标/QC/划分/尺度一致，仅允许训练FM预测不同。
只写新的study、初始化权重副本和小型cache_view.json；原npz、原study、部署包全部只读。
成功标志：ready=true、train_records=156、cache_regenerated=false、optimizer_updates=0，三个组辅助权重分别为0/0/0.1。

缓存仍保留40个验证目标和92个配对，这是存储范围；每次评分及选择best前排除不可靠I，不需要删除缓存：

|任务/设备|评分配对数|唯一目标数|
|---|---:|---:|
|Task1|27|21|
|Task2机器|39|39|
|Task2体脂秤|24|24|

排除两个配对指向的同一目标d12_f2efb4a4c95a40。异常目标导联继续按原QC mask排除，不删除其他正常导联。
三组第0轮均应复现r1=0.70235417、r2=0.69846904、平均r=0.70041161，容差2e-5；不符会在第一步优化器更新之前停止。
prepare成功后不再重复执行。预检失败时保留日志，用新的输出名排查重试，不覆盖旧study或改SHA绕过校验。

## 6. 三个终端分别启动GPU0、GPU1、GPU2

终端A：原FM训练缓存＋原损失，GPU0。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc_ms

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --group control_old_fm \
  --gpu 0
```

终端B：control100训练缓存＋原损失，GPU1。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc_ms

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --group adapt_control100 \
  --gpu 1
```

终端C：control100训练缓存＋分尺度辅助损失，GPU2。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc_ms

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --group adapt_multiscale \
  --gpu 2
```

这是三个独立终端；在同一终端依次粘贴会顺序执行，不是并行。
每个脚本依次完成第0轮验证、20轮慢CNN训练、best/last完整评估及best便携模型导出。
三组只更新慢CNN，验证FM快残差读相同control100缓存，采样仍对应Heun16/K16，β=0.75。
新增多尺度项只用于C组训练，不在验证分数中加权，也不会改变测试输入与网络。

## 7. 中断续跑

不重新prepare、不重新生成缓存。以C组为例：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc_ms

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --group adapt_multiscale \
  --gpu 2 \
  --resume /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1/adapt_multiscale/last.pt
```

A/B对应修改group、GPU及resume目录。只允许本轮同组、同study、同输出目录、同辅助项配置的精确续跑。
旧两组partner_v1或旧SlowPCC checkpoint不能作为本轮resume。PCC20已经是共同初始化，不需要另传旧last。

## 8. 三组结束后汇总

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc_ms

python -m baselines.B5_control_pcc.compare \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1
```

输出在partner_multiscale_v1/comparison.csv和comparison.json，包含三组best/last共六项、相对第0轮变化，以及best/last各自的B-A、C-B。
先检查第0轮相同，再看同轮曲线、best/last、逐导联与逐患者变化。
若C改善，则支持这套辅助监督在当前搭档/预算下有效，不代表每个尺度都同样受益；还要结合新增损失日志、波形图与后续配对seed验证。
若无改善，优先看辅助项是否实际生效、是否损害局部形态，而不是直接增大权重。
只看完整ECG两任务平均r选best，RMSE和慢r作为诊断。best仍可能是第0轮，表示没有超过初始化。
原损失与分尺度损失都下降也不保证验证r提高，最终以同QC范围下的实际组合波形评分为准。

## 9. 打包报告

```bash
cd /home/qht/huawei/results/B5-best-exp-Control100PCC

tar --exclude='*.pt' --exclude='*.npz' --exclude='*.npy' \
  -czf /home/qht/huawei/B5_control100_multiscale_reports.tar.gz \
  partner_multiscale_v1
```

下载/home/qht/huawei/B5_control100_multiscale_reports.tar.gz即可，不需要发送原始缓存或模型权重。

## 10. 三组已完成后：先做离线诊断

诊断第0轮、B20、C20的分尺度误差和逐患者变化，见[诊断与并行复核操作](b5_control100_multiscale_diagnosis.md)。
该入口只读取已有数组，不重新训练；患者成组bootstrap可同时完成，训练seed复核则需要独立实验配置。
