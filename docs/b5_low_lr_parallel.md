# B5低学习率三组实验：服务器完整命令

本轮仍使用B5-best-exp-ScaleAblation分支，新增--learning-rate参数。本地工作树：
`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-ScaleAblation`。

服务器新建`/home/qht/huawei_B5_scale_lowlr`，不更新旧实验worktree，不更改旧冻结study。
三组从同一个原始PCC20初始化：B原损失、L长尺度、ALL三尺度；学习率统一2e-5，训练seed43、20轮、每5轮验证。
其余配置及初始化均与此前1e-4对应组一致。推理seed42、Heun16/K16、β=0.75，FM冻结，按最终组合平均r选best。
准备入口仍生成四组能力的study，但本轮只启动B/L/ALL，不启动SM；这不会改变三组结果。

## 1. 本地推送

```powershell
cd "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-ScaleAblation"
git push origin B5-best-exp-ScaleAblation
```

按原有流程同步Gitee后，再执行以下服务器命令。

## 2. 创建服务器新worktree，只做一次

```bash
conda activate BioFlow
cd /home/qht/huawei

git fetch gitee B5-best-exp-ScaleAblation
git worktree add --detach /home/qht/huawei_B5_scale_lowlr gitee/B5-best-exp-ScaleAblation

cd /home/qht/huawei_B5_scale_lowlr
git log -1 --oneline
python -m baselines.B5_control_pcc.prepare --help
```

help应能看到--learning-rate。新目录创建后保留本次代码，整个实验期间不更新核心代码。

## 3. 准备新study：一次，不训练

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_scale_lowlr

run=/home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_lr2em5_v1
mkdir -p "$run"

python -u -m baselines.B5_control_pcc.prepare \
  --old-run /home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1 \
  --control100-run /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1 \
  --reference-report /home/qht/huawei/results/B5-rescore-reliable-I/default_20261017_192455/correlation_score.json \
  --training-seed 43 \
  --learning-rate 2e-5 \
  --epochs 20 \
  --multiscale-weight 0.1 \
  --output "$run/shared" \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

成功后确认ready=true、learning_rate=2e-05、training_seed=43、inference_seed=42。
这里只校验/复用已有缓存，并复制同一PCC20初始化，不会从C20或L20续训。
不要重做已完成的prepare，不编辑旧seed43_v1/study或手工改SHA。

## 4. 三张GPU三个终端，准备完成后可同时启动

终端一：GPU0，B-low。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_scale_lowlr

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_lr2em5_v1 \
  --group adapt_control100 \
  --gpu 0
```

终端二：GPU1，L-low。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_scale_lowlr

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_lr2em5_v1 \
  --group adapt_long \
  --gpu 1
```

终端三：GPU2，ALL-low。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_scale_lowlr

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-ScaleAblation/seed43_lr2em5_v1 \
  --group adapt_multiscale \
  --gpu 2
```

每组自动完成训练、best/last评估、best组合包导出。显卡应分别空闲再启动；脚本是前台任务，可在三个独立tmux会话中分别执行，避免SSH断开中断训练。

本轮保持与上轮相同验证节点0/5/10/15/20，不额外改变验证频率。判断时同时比较固定20轮、best及Task2是否减少退步。

## 5. 三组均完成后打包

```bash
tar -czf /home/qht/huawei/B5_low_lr_seed43_reports.tar.gz \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-best-exp-ScaleAblation \
  seed43_lr2em5_v1
```

下载路径：`/home/qht/huawei/B5_low_lr_seed43_reports.tar.gz`。
服务器保留全部原始权重和数组，压缩包只排除大文件，保留评分、日志、study及初始化等审计信息。
