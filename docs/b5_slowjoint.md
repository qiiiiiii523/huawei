# 局部慢CNN＋G120共同训练：H/P

分支 `B5-final-exp-slowjoint`，从已完成G120的 `68b4db5` 创建。保留原S50窗口、G120两层注意力和输出相加，不增加中间融合。原FM、历史context处理、β=0.75、Heun16/K16/噪声seed42、120秒及QC评分规则固定。

两组从同一份原P慢CNN和已训练G120-best初始化：

|组|训练流程|
|---|---|
|H|华为共同训练20轮|
|P|PTB真实120秒记录共同训练20轮，固定使用公开最后一轮，再重置优化器，华为共同训练20轮|

共同训练的是P慢CNN和G120，监督两者相加后的总慢预测。华为阶段沿用原慢损失及固定FM组合后的PCC；训练损失不经过历史context，周期选模经过原Final历史context。公开阶段只有总慢监督，不生成公开FM。两组华为阶段相同数据顺序seed43、record batch2、P学习率1e-5、G学习率1e-4、每5轮验证。P额外包含公开计算量，不宣称总训练成本相同。第0轮和P公开训练后/华为第0轮都是候选；仅保留 `best_r.pt`、`last.pt`、训练历史。

## 1. 服务器代码目录

在原仓库执行（本地提交后需先自行推送分支）：

```bash
cd /home/qht/huawei
git fetch gitee B5-final-exp-slowjoint
git worktree add --detach /home/qht/huawei_slowjoint FETCH_HEAD
cd /home/qht/huawei_slowjoint
conda activate BioFlow
```

原结果路径保持不动。缓存读取原G120的 `shared/`，不会重跑FM，也不会复制原完整缓存。

## 2. 只准备一次

以下源run为已完成的seed43 G120实验；先检查文件存在。若实际另存了G120最佳权重，应明确替换对应路径，不能换成原B5的best.pt。

```bash
cd /home/qht/huawei_slowjoint
root=/home/qht/huawei/results/B5-final-exp-slowjoint/joint_v1
source_run=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863
mkdir -p "$root"
test -f "$source_run/shared/study.json"
test -f "$source_run/G120/best.pt"
```

准备公开数据（CPU，无训练）。仅使用真正达到120秒的PTB Diagnostic；115.2秒记录排除，不做补长。它比旧50～120秒公开缓存小，实际患者数与排除原因见audit.json/csv。按患者划分后再筛长度，沿用公开slow的split seed42。

```bash
set -o pipefail
python -u -m baselines.B5_slowjoint.public \
  --root /home/qht/huawei/data/PTB_db \
  --bundle /home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --output "$root/public120" \
  2>&1 | tee "$root/public_prepare.log"
```

数据若放在 `data/PTB-Diagnostic`，只替换上面的 `--root`。H单独实验可以跳过公开准备，并在下条命令中省略 `--public-cache`；这种shared不能运行P。

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u -m baselines.B5_slowjoint.prepare \
  --source-run "$source_run/shared" \
  --g120-checkpoint "$source_run/G120/best.pt" \
  --public-cache "$root/public120" \
  --output "$root/shared" --device cuda \
  2>&1 | tee "$root/prepare.log"
```

必须出现 `ready: true`。预检用完整验证流程复现源G120，允许分批卷积造成不超过2e-6的r差异；不是用随机数据代替成绩复现。准备不更新参数。中断准备用新输出目录，不删除原缓存。

## 3. 两个终端分别训练与评估

终端A（GPU0）：

```bash
cd /home/qht/huawei_slowjoint
conda activate BioFlow
bash scripts/run_b5_slowjoint.sh --group H --gpu 0
```

终端B（GPU1）：

```bash
cd /home/qht/huawei_slowjoint
conda activate BioFlow
bash scripts/run_b5_slowjoint.sh --group P --gpu 1
```

脚本自动完成训练和best_r/last复评；中断后原命令可从本组last.pt续跑。若训练已完成，仅补评估。共享缓存只读；两组写不同目录。无需在脚本中拉取Git。

## 4. 汇总与打包

```bash
cd /home/qht/huawei_slowjoint
root=/home/qht/huawei/results/B5-final-exp-slowjoint/joint_v1
python -m baselines.B5_slowjoint.summarize --run "$root"
cd /home/qht/huawei
find results/B5-final-exp-slowjoint/joint_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \) \
  -print0 | tar --null -T - -czf B5_slowjoint_reports.tar.gz
```

只完成H时汇总加 `--groups H`。比较的是最终 `(r1+r2)/2`，RMSE不参与选模。没有超过源G120则继续保留源权重。

## 5. 新记录推理

`baselines.B5_slowjoint.predict`接收 `--run shared目录 --checkpoint 本组/best_r.pt --anchor-npy 新I.npy --record-key 新记录ID --task-id task1 --input-type watch_ecg --output 新输出目录`，另支持人口学JSON和合法历史数组。只读取study.json及模型权重，不需要训练缓存、真实目标或QC标签。I必须是原始μV、500Hz、120秒；历史仍按原Final接口处理。
