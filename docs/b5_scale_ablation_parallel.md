# B5训练seed复核、尺度消融、边缘诊断：并行操作

## 1. 本轮只需要四组训练

分支：`B5-best-exp-ScaleAblation`。

本地工作树：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-ScaleAblation`。

服务器新工作树：`/home/qht/huawei_B5_scale_ablation`。

旧工作树与旧study保持原样。本分支训练协议升级到v3，不能拿旧v2 study在新训练入口续训。

|简称|命令group|新增监督|训练seed|用途|
|---|---|---|---|---|
|B|adapt_control100|无|43|原损失对照|
|SM|adapt_short_middle|短＋中尺度|43|尺度消融|
|L|adapt_long|长尺度|43|尺度消融|
|ALL|adapt_multiscale|三尺度|43|对应上轮C，完成seed复核|

新B/ALL的差异，与旧seed42 B/C的差异进行比较，就是训练seed复核。无需额外再训练两组。
SM/L/ALL共享辅助损失总系数0.1，对启用的尺度取平均；原损失全部保留。它检验固定总辅助权重下的监督分配，不是保持单尺度系数不变的严格删除实验。

四组从同一个**原始PCC20慢CNN**初始化，来自现有control100缓存运行中的shared/components.pt；不是从刚训练的C20初始化。
全部使用control100训练/验证FM缓存。FM不训练，不重新推理，不生成新缓存。
四组共同20轮、原学习率/损失基础项、每5轮验证，按最终120秒组合平均r选best，保留第0轮候选。
推理仍是seed42、Heun16/K16、β=0.75、可靠I＋目标导联QC。新的seed43只用于训练随机性。

边缘诊断在CPU上读取**旧seed42的第0轮/B20/C20**，可与四组训练并行，输出单独存放。

## 2. 推送和创建服务器worktree

本机PowerShell推送：

```powershell
cd "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-ScaleAblation"
git push origin B5-best-exp-ScaleAblation
```

按原有流程同步到Gitee后，在服务器执行以下命令。这里只新建worktree，不切换/home/qht/huawei当前分支，不移动.gitattributes等文件。

```bash
conda activate BioFlow
cd /home/qht/huawei

git fetch gitee B5-best-exp-ScaleAblation
git worktree add --detach /home/qht/huawei_B5_scale_ablation gitee/B5-best-exp-ScaleAblation

cd /home/qht/huawei_B5_scale_ablation
git log -1 --oneline
```

创建只执行一次。如果目录已经是这个实验的worktree，跳过创建，改为：

```bash
cd /home/qht/huawei_B5_scale_ablation
git status --short
git fetch gitee B5-best-exp-ScaleAblation
git merge --ff-only gitee/B5-best-exp-ScaleAblation
```

新study准备后不要更新其训练核心代码；需要改代码时为下一轮建立新study。旧结果的只读诊断不执行旧模型加载器，不受训练指纹限制。

## 3. 准备新study：先执行一次，不训练

需要既有两个源cache和上次可靠I重评分文件。它们仅作为只读来源；新的准备入口复制原PCC20组件并冻结seed43配置、尺度映射和训练集阈值。

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_scale_ablation

run=/home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_v1
mkdir -p "$run"

python -u -m baselines.B5_control_pcc.prepare \
  --old-run /home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1 \
  --control100-run /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1 \
  --reference-report /home/qht/huawei/results/B5-rescore-reliable-I/default_20261017_192455/correlation_score.json \
  --training-seed 43 \
  --epochs 20 \
  --multiscale-weight 0.1 \
  --output "$run/shared" \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

成功标志：ready=true、training_started=false、optimizer_updates=0、training_seed=43、inference_seed=42，四组enabled_bands分别为[]、[0,1]、[2]、[0,1,2]。
四组训练源均为control100缓存，初始化SHA相同。首次验证会再次核对原始平均r约0.7004116。
准备过程主要校验磁盘缓存，CPU前向不占GPU。已成功则不要重复prepare；不要在shared里面tee日志，否则输出会被判断非空。

## 4. 三张GPU：三个终端分别启动

每个终端激活BioFlow。下面命令是前台训练，使用三个独立tmux会话或各自稳定的终端；脚本本身不会自动脱离SSH。
三组可同时启动，GPU2完成SM后自动串行启动L，共四组训练。不要把L另起一次重复跑。

终端A，GPU0跑B：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_scale_ablation

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_v1 \
  --group adapt_control100 \
  --gpu 0
```

终端B，GPU1跑ALL：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_scale_ablation

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_v1 \
  --group adapt_multiscale \
  --gpu 1
```

终端C，GPU2先SM后L：

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_scale_ablation
run=/home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_v1

bash scripts/run_b5_control100_pcc.sh \
  --run "$run" \
  --group adapt_short_middle \
  --gpu 2

bash scripts/run_b5_control100_pcc.sh \
  --run "$run" \
  --group adapt_long \
  --gpu 2
)
```

每组脚本自动完成训练、best/last正式评估、导出best组合包。日志在group目录外，不会触发非空训练目录报错。
若SM失败，该终端会停止而不会启动L；先查看对应.train.log。续训同组加--resume绝对路径/last.pt，不能重做prepare或更改已冻结seed。

## 5. 第四个终端：CPU边缘诊断，可与训练并行

不需要GPU，不重新生成FM。旧波形数组、缓存必须还在，只有上次上传的CSV包不能恢复这个诊断。

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_scale_ablation

out=/home/qht/huawei/results/B5-best-exp-ScaleAblation/edge_v1
mkdir -p "$(dirname "$out")"

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u scripts/diagnose_b5_record_edges.py \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --output "$out" \
  2>&1 | tee "${out}.log"
)
```

只执行一次；重跑改edge_v2等新目录，不覆盖已有结果。诊断读取旧study，因此也可以在四组训练开始前运行。

主要输出：

|文件|回答什么|
|---|---|
|region_summary.csv|首20秒、中间80秒、末20秒是否存在系统差异；每个评分集合对导联等权的诊断指标|
|target_lead_regions.csv、lead_regions.csv|哪些患者/导联在边缘失败；重复配对先按目标平均|
|pair_lead_regions.csv|原始配对级诊断|
|S50_context_layout.csv|12窗真实输入、前后重复补边、中心裁剪范围|
|anchor_regions.csv|输入I在三个区域的归一化均值/起伏强度，含明确QC列|
|edge_audit.json、qc_exclusions.csv|来源校验、只读状态和质量排除|

S50首窗输入范围为−20～30秒，左边20秒重复记录首值；第二窗左边重复10秒；末两窗对称。每窗只输出中间10秒，整条输出仍是120秒真实时间范围，无额外补齐输出参与正式评分。
已用序列探针测试：中央裁剪和拼接恢复原始0～120秒顺序，训练可微generate与部署predict_native相同。
这个测试验证接口，不证明真实边缘预测一定好；边缘较差也可能来自真实目标波动较大或输入信息不足。

区域shape误差是区域内“慢预测−完整目标MA501”的标准差，只在诊断中去除固定偏移，不改预测、不在测试时扣真值均值。
中段80秒与边缘20秒长度不同，区域r不能直接代替正式r；结合形状误差、真实起伏强度和患者案例解释。
region_summary里不可靠I已排除；anchor_regions保留输入审计明细并用input_I_QC_good区分，不把坏I当正常记录平均。

## 6. 全部结束后打包

确认四组训练和边缘诊断都完成，再执行：

```bash
cd /home/qht/huawei

tar -czf /home/qht/huawei/B5_scale_ablation_seed43_reports.tar.gz \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-best-exp-ScaleAblation \
  seed43_v1 edge_v1 edge_v1.log
```

下载：`/home/qht/huawei/B5_scale_ablation_seed43_reports.tar.gz`。

包里保留study、cache_view、preflight、初始化、损失历史、验证评分和边缘表格，不包含大权重/预测数组；服务器原文件都保留。
后续先比较seed43的ALL−B是否重复seed42方向，再比较SM/L/ALL在同seed下的差异；同时检查Task2和逐导联取舍。两个seed仍不足以证明普遍稳定，但比单次结果更有依据。
