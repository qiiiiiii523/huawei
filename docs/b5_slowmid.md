# B5-final-exp-slowmid：让全局信息进入慢CNN内部

## 本轮只改变什么

本分支独立于原G120、SlowFusion及纯共同训练分支，不改原实验文件。复用已完成G120的原P权重与G120-best，固定FM、历史context、β=0.75、Heun16/K16/噪声seed42、可靠I与目标导联QC、120秒和最终 `(r1+r2)/2`。

**ADD是共同训练对照，不是冻结P的旧G120；MID才增加中间特征融合。** 两者从完全相同的P、G120权重起步，共同更新原网络。原G120成绩仍作为参考，保留第0轮候选。没有任何已验证的提升承诺。

```text
当前120秒I → 抗混叠5Hz → 原G120两层注意力
                              ├→ 原11导联修正输出 ─────────────────┐
                              └→ 32通道全局隐藏特征                │
                                       ↓ 对齐到50Hz并切对应50秒    │
I的50秒窗口 → 原慢CNN块1～6 → 零初始化的时间FiLM → 原块7～8 → 输出 ─┤
                                                                   ↓
                                                               总慢预测
                                                                   ↓
                          固定FM快残差＋0.75×总慢预测→原context→完整输出
```

G120不只输出一条修正曲线，而是把**输出投影之前**的全局特征送入局部CNN。第一处融合放在第6块之后，给后面dilation64/128的两块继续处理；这是起始设计，不是已证实的最佳位置。

融合是每个时间位置、每个隐藏通道的条件调节：`h' = h × (1 + 0.1×tanh(γ)) + β`。γ、β来自全局特征的通道LayerNorm和两个1×1卷积；不沿时间去均值。最后投影权重和偏置全零，因此初始化是恒等映射；原G120输出相加路径保留。32通道配置新增4288参数，避免再加一套大注意力/六个导联专用头。梯度同时更新P、G120、融合模块。

与旧SlowFusion不同：旧版冻结P，利用最终局部特征额外预测残差；本轮真正修改P内部计算，并允许P适应全局信息。仍不把慢标签硬拆成两个频段。

## 先跑哪两组

|组|华为训练|区别|
|---|---|---|
|H_add|20轮，P＋G120共同更新|只在输出相加，匹配对照|
|H_mid|相同20轮，P＋G120＋融合共同更新|第6块后时间FiLM|

record batch2、seed43、P学习率1e-5、G120及融合学习率1e-4、AdamW/weight decay0.01、梯度裁剪1、每5轮验证。两组数据顺序、损失、可见输入相同；MID增加了4288参数，因此结果代表这套融合结构的整体效果，不宣称完全分离容量因素。

损失监督P＋G的**总慢预测**，沿用原Huber、慢PCC、振荡残差及固定FM组合PCC，不让两条路径各自拟合完整慢标签再相加。华为训练损失沿用不经过context的基础组合，选模和正式复评都经过固定的原context。RMSE只作诊断，不参与选模。

## 1. 服务器建立代码目录

本地已提交，需先自行推送 `B5-final-exp-slowmid`。服务器：

```bash
cd /home/qht/huawei
git fetch gitee B5-final-exp-slowmid
git worktree add --detach /home/qht/huawei_slowmid FETCH_HEAD
cd /home/qht/huawei_slowmid
conda activate BioFlow
```

## 2. 一次准备与初始成绩复现

复用已有原G120缓存，不重跑FM，也不复制旧缓存。这里的源checkpoint是**G120的best.pt**，不是原B5的best.pt。

```bash
(
set -euo pipefail
cd /home/qht/huawei_slowmid
root=/home/qht/huawei/results/B5-final-exp-slowmid/mid_v1
source_run=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863
mkdir -p "$root"

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u -m baselines.B5_slowjoint.prepare \
  --source-run "$source_run/shared" \
  --g120-checkpoint "$source_run/G120/best.pt" \
  --output "$root/shared" --device cuda \
  2>&1 | tee "$root/prepare.log"
)
```

须出现 `ready: true`。服务器预检实际重算ADD、MID完整验证成绩，两者初始化应复现当前G120（r容差2e-6）；检查不通过不生成可训练study。准备用新目录，中断不删除原资产。相同旧缓存须能从该代码分支读取；原数据、配置、预测数组和历史权重都不能在期间变更。

## 3. 两个终端并行

终端A：

```bash
cd /home/qht/huawei_slowmid
conda activate BioFlow
bash scripts/run_b5_slowmid.sh --group H --mode add --gpu 0
```

终端B：

```bash
cd /home/qht/huawei_slowmid
conda activate BioFlow
bash scripts/run_b5_slowmid.sh --group H --mode mid --gpu 1
```

脚本训练后自动复评实际 `best_r.pt`、`last.pt`。中断后原命令从本组last.pt续跑；已完成训练只补评估。结果分别在 `mid_v1/H_add`、`mid_v1/H_mid`。无需手动改Python命令、无需在训练中切Git分支。若用新run，准备和两组脚本都传一致的 `--run` 路径。

## 4. 汇总和打包

```bash
cd /home/qht/huawei_slowmid
root=/home/qht/huawei/results/B5-final-exp-slowmid/mid_v1
python -m baselines.B5_slowjoint.compare_mid --run "$root"

cd /home/qht/huawei
find results/B5-final-exp-slowmid/mid_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \) \
  -print0 | tar --null -T - -czf B5_slowmid_reports.tar.gz
```

看 `comparison/summary.md`：MID既要超过旧G120，也要超过同预算ADD。逐患者、逐导联和慢曲线分尺度统计用于看提升分布；成绩很小的差异再做第二训练seed，不直接视为稳定收益。

## 5. 可选：公开阶段就学习融合

P路线已支持，但第一轮不必四组全跑。若要P，先准备一个**新的**shared，绑定公开缓存；不能给现有不可变shared事后追加公开数据。

```bash
(
set -euo pipefail
cd /home/qht/huawei_slowmid
root=/home/qht/huawei/results/B5-final-exp-slowmid/mid_public_v1
source_run=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863
mkdir -p "$root"
python -u -m baselines.B5_slowjoint.public \
  --root /home/qht/huawei/data/PTB_db \
  --bundle /home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --output "$root/public120" 2>&1 | tee "$root/public_prepare.log"

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u -m baselines.B5_slowjoint.prepare \
  --source-run "$source_run/shared" --g120-checkpoint "$source_run/G120/best.pt" \
  --public-cache "$root/public120" --output "$root/shared" --device cuda \
  2>&1 | tee "$root/prepare.log"
)
```

只纳入真实>=120秒PTB Diagnostic，固定患者划分、尺度、绝对电压和QC。公开阶段20轮只监督总慢曲线，不生成公开FM；固定公开last进入华为20轮并重置优化器。公开验证仅诊断，最终仍按华为完整组合r选模，保留原始G120第0轮候选。

两个终端分别执行：

```bash
bash scripts/run_b5_slowmid.sh --run /home/qht/huawei/results/B5-final-exp-slowmid/mid_public_v1 --group P --mode add --gpu 0
```

```bash
bash scripts/run_b5_slowmid.sh --run /home/qht/huawei/results/B5-final-exp-slowmid/mid_public_v1 --group P --mode mid --gpu 1
```

汇总时 `--arms P_add P_mid`。P比H多了公开训练计算量。PTB实际120秒记录较少，以audit统计为准；不是几万条长记录。

## 6. 能否测试新记录

可以。检查点保存P、G120及融合模块的权重和模式。使用 `baselines.B5_slowjoint.predict`，按checkpoint自动选择ADD或MID；只读取study.json和权重，不读取训练缓存/真实目标/真实慢曲线。例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.B5_slowjoint.predict \
  --run /home/qht/huawei/results/B5-final-exp-slowmid/mid_v1/shared \
  --checkpoint /home/qht/huawei/results/B5-final-exp-slowmid/mid_v1/H_mid/best_r.pt \
  --anchor-npy /path/to/new_I_uV.npy --metadata-json /path/to/new_metadata.json \
  --record-key new_record_001 --task-id task1 --input-type watch_ecg \
  --output /home/qht/huawei/results/B5-final-exp-slowmid/new_record_001 --device cuda
```

Task2合法历史仍走原Final接口。本轮没有新增历史context编码器，“全局特征”来自当前120秒I，不是历史心电。
