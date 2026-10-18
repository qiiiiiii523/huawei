# P路线：部署复现、加密验证、学习率比较

## 1. 当前结论与本轮任务

当前相同QC评分范围内，原最佳平均r=0.700412；P路线在训练seed42/43下分别达到0.705124/0.705094。**公开长记录适配慢CNN，再用华为微调，这条完整路线有重复的正收益。**

这里仍不使用额外设备context；慢CNN本身继续使用50秒同步I。FM不训练、不改网络，固定control100；Heun16/K16、推理噪声seed42、β=0.75、评分集合不变。训练seed42/43与推理噪声seed是两回事。

本轮推进P，M现有权重保留，不重复训练M。完成三个任务：

| 任务 | 操作 | 是否训练 |
|---|---|---|
| ① 复现部署 | 两份已有P组合包都从噪声实际生成、完整120秒validate，对照各自报告 | 否；GPU2顺序执行 |
| ② 加密选模 | 两个训练seed，各复用自己的P公开阶段末尾权重，重跑华为阶段；每78次更新验证直到780次 | 是；只重跑华为阶段 |
| ③ 学习率对照 | 每个seed比较1e-4与3e-5；初始化、数据顺序、损失、780次预算、验证点一致 | 与②合成四组，不再额外重跑一套 |

**不需要重新下载PTB，不需要重新处理数据，不需要重新训练公开阶段。旧目录和旧模型保留，新实验写入P_refine_v1。**

## 2. 本地与服务器代码位置

```text
分支：B5-best-exp-PublicSlow
本地：C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-PublicSlow
服务器：/home/qht/huawei_B5_public_slow
服务器数据/旧结果：/home/qht/huawei
```

先将本地该分支的新提交推送并同步到服务器使用的Gitee远程。然后服务器执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow
git fetch gitee B5-best-exp-PublicSlow
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

若工作树有本地修改导致拒绝更新，先保留修改，不使用强制reset。已上传的模型和结果不需要移动到代码工作树。

只复制代码框内命令，不复制三个反引号。

## 3. 先做①：GPU2实际生成，复核两份已有P

```bash
cd /home/qht/huawei_B5_public_slow
bash scripts/check_b5_public_slow_p.sh --gpu 2
```

脚本内部顺序验证：

```text
seed42：/home/qht/huawei/results/B5-slow-public/CMP_v1/P/model.pt
seed43：/home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1/P/model.pt
```

读取数据配置：

```text
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
```

这一步使用真正的FM生成，不复用训练快残差缓存。GPU会有实际推理计算，耗时比缓存评分长。每份包先核对导出哈希和P组身份，然后由B5-best validate核对数据来源与评分参考。

成功时打印`Both P packages reproduced`，两行`reference_matched`均为True。若失败，保留错误和evaluation.json，先排查，暂不更新稳定B5-best。

结果自动放到新目录，最后会打印实际路径：

```text
/home/qht/huawei/results/B5-slow-public/P_deployment_时间戳_进程号/
    summary.csv
    summary.json
    seed42/evaluation.json
    seed43/evaluation.json
```

## 4. 准备②③：CPU执行一次，不训练

先确认两份公开阶段末尾权重还在：

```bash
ls -lh /home/qht/huawei/results/B5-slow-public/CMP_v1/P/stage1_last.pt
ls -lh /home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1/P/stage1_last.pt
```

需要的是`stage1_last.pt`，**不是**P的`best.pt`或`model.pt`。本轮从这里重新开始华为阶段，才能检查原学习率复现并公平比较学习率。

准备四组：

```bash
cd /home/qht/huawei_B5_public_slow
python -u -m baselines.B5_public_slow.refine prepare \
  --reference42 /home/qht/huawei/results/B5-slow-public/CMP_v1 \
  --reference43 /home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1 \
  --output /home/qht/huawei/results/B5-slow-public/P_refine_v1
```

这会核对旧study、原始初始化、每份阶段一权重的所属seed/阶段/FM身份、缓存数据哈希、患者划分及当前训练与部署前向兼容性。大量文件会先在CPU上读取，所以此时GPU不动是正常的。

旧study的历史代码哈希会记录在新study中，不会修改旧study，也不会绕过旧study的续跑保护。每组创建新的当前代码study，并把旧文件哈希列为依赖。

最后打印`Prepared four stage2-only trials`，才表示准备完成。该步骤只执行一次；目录存在且非空时拒绝覆盖。若准备中断或需要另开实验，改用新输出目录，并将后续`--root`统一指向新目录。

## 5. 两个终端并行：②③一起完成

四组如下。每个seed内部的两份公开阶段权重完全相同，只改变学习率；两个seed分别使用各自的公开阶段权重。

| 组 | 训练seed | 阶段二学习率 | 更新次数 | 验证 |
|---|---:|---:|---:|---|
| seed42_lr1e4 | 42 | 1e-4 | 780 | 起点及每78次 |
| seed42_lr3e5 | 42 | 3e-5 | 780 | 相同 |
| seed43_lr1e4 | 43 | 1e-4 | 780 | 相同 |
| seed43_lr3e5 | 43 | 3e-5 | 780 | 相同 |

780次约10轮华为记录遍历，78次约1轮。两个学习率均使用新建的阶段二AdamW，不继承公开阶段的优化器动量，和原阶段切换规则一致。不是从当前P-best继续训练。

**终端A：GPU0，seed42的两种学习率顺序完成。**

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow
bash scripts/run_b5_public_slow_refine.sh --seed 42 --gpu 0
```

**终端B：GPU1，seed43的两种学习率顺序完成。**

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow
bash scripts/run_b5_public_slow_refine.sh --seed 43 --gpu 1
```

无需加后台`&`，分别在两个终端运行即可。使用tmux等持久终端可以避免关闭SSH造成中断。

每个seed先跑1e-4组，自动检查第390次更新能否复现旧成绩，误差容限2e-5：

| seed | 旧第390次平均r |
|---|---:|
| 42 | 0.705123795 |
| 43 | 0.705094488 |

检查通过后再跑3e-5。若复现失败，脚本停止，输出该组`reproduction.json`供分析；不把不一致的运行当成学习率收益。

每组都保存：

- `P/best.pt`：共同旧起点、阶段二开始点及每78次更新中平均r最高的权重。
- `P/last.pt`：最新训练状态，支持续跑。
- `P/stage2_last.pt`：第780次更新状态。
- `P/model.pt`：完成后自动导出best对应的便携组合包，固定FM＋该慢CNN。

本轮`best.pt`直接包含全部新验证点；不再将加密高点排除。旧seed42/43稀疏选模结果保持原样，不能把新候选池更大的收益全部解释成训练方法收益。

中断后在原命令末尾加`--resume`。同一个seed中已开始的组从last续跑，未开始的组正常启动，例如：

```bash
bash scripts/run_b5_public_slow_refine.sh --seed 42 --gpu 0 --resume
```

## 6. 两个终端都结束后，汇总四组

```bash
cd /home/qht/huawei_B5_public_slow
run=/home/qht/huawei/results/B5-slow-public/P_refine_v1
python -m baselines.B5_public_slow.refine summarize \
  --root "$run" --output "$run/comparison"
cat "$run/comparison/summary.csv"
```

重点比较：

1. 原1e-4在相同训练轨迹中是否找到比390次更好的点：判断加密选模收益。
2. 同一个seed、同一网格下3e-5是否比1e-4更好：判断学习率收益。
3. 两个seed的改善方向是否一致，而非只挑一份最高分。
4. Task1、Task2及逐导联是否有明显退化。RMSE保留作诊断，不参与best选择。

`comparison/curves.csv`包含四组全部阶段二验证点，便于检查高点位置与后续回落。汇总也会重新核对同seed权重来源、各组数据/损失身份与1e-4复现情况。

若`comparison`已存在，不覆盖它；另用`comparison_v2`等新名称。

## 7. 打包报告给我

不打包权重、原始数据或预测大数组。先打包训练与学习率比较：

```bash
run=/home/qht/huawei/results/B5-slow-public/P_refine_v1
tar --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  -czf /home/qht/huawei/B5_public_slow_P_refine_reports.tar.gz \
  -C "$run" trials.json seed42_lr1e4 seed42_lr3e5 \
  seed43_lr1e4 seed43_lr3e5 comparison
```

服务器打包文件：`/home/qht/huawei/B5_public_slow_P_refine_reports.tar.gz`。

部署结果另打包，`out`使用第3步最后打印的实际目录：

```bash
out=/home/qht/huawei/results/B5-slow-public/P_deployment_替换为实际后缀
tar --exclude='*.npy' --exclude='*.npz' \
  -czf /home/qht/huawei/B5_public_slow_P_deployment_reports.tar.gz \
  -C "$out" .
```

服务器文件：`/home/qht/huawei/B5_public_slow_P_deployment_reports.tar.gz`。

完成后先分析结果，再决定是否晋升新的B5-best默认包。本轮不自动修改稳定分支。
