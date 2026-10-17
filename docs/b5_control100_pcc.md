# Control100慢CNN适应实验：复用已完成缓存，采用可靠I验证

## 本轮只验证这一件事

目前部署是control100 FM快残差＋PCC20慢CNN，β=0.75。PCC20此前针对原FM快残差训练。
本轮从同一PCC20初始化两组慢CNN，判断改用control100训练快残差，能否更适合当前部署组合。

|设置|control_old_fm 对照|adapt_control100 实验|
|---|---|---|
|慢CNN初始化|同一PCC20|同一PCC20|
|训练组合PCC中的固定快残差|原FM缓存|control100缓存|
|慢标签、F监督、组合PCC及系数|原设置|完全相同|
|网络、输入、β、学习率、优化器、随机种子、预算|相同，继续20轮|相同，继续20轮|
|最终验证组合|control100＋该组慢CNN|control100＋该组慢CNN|
|验证与选best|可靠I＋目标导联QC，完整120秒，两任务平均r|相同|

这里的control仍然有组合PCC，不是之前“无组合PCC”的control组。
两组都不更新FM、不重新生成FM缓存、不改变FM训练路径，不使用验证患者训练。
最终都配control100评估，才能隔离训练搭档适配的作用。不能拿旧FM验证分和新FM验证分直接比较来判断适配是否有效。

## 1. 本地代码与推送

```text
分支：B5-best-exp-Control100PCC
基于最新B5-best ec12008，含可靠I验证。
工作树：C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Control100PCC
```

本机PowerShell：

```powershell
cd "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Control100PCC"
git push origin B5-best-exp-Control100PCC
```

本机origin是GitHub；服务器使用Gitee。先按原有镜像流程同步该分支到Gitee。
旧Best-SlowPCC和B5-best稳定工作树保持不变，不把新核心复制进旧冻结实验。

## 2. 服务器创建独立工作树（首次）

```bash
conda activate BioFlow
cd /home/qht/huawei
git fetch gitee B5-best-exp-Control100PCC
git worktree add --detach /home/qht/huawei_B5_control100_pcc FETCH_HEAD
cd /home/qht/huawei_B5_control100_pcc
git log -1 --oneline
```

已有该目录时，在该目录执行fetch及`git merge --ff-only FETCH_HEAD`，不要重复worktree add。

## 3. 只读检查和创建新实验：CPU，不重新生成缓存

本轮直接读取以下已经完成的来源：

```text
原FM训练缓存：
/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1/cache

control100训练及验证缓存（你刚完成的）：
/home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache

同一control100＋PCC20初始化：
control100_pcc20_cache_v1/shared/components.pt

你已重评分的新参考：
/home/qht/huawei/results/B5-rescore-reliable-I/default_20261017_192455/correlation_score.json
```

不需要再运行旧prepare或cache命令。以下命令会校验两份缓存、权重、目标、输入I、人口学、QC、划分和SHA；只允许训练快残差不同。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_control100_pcc

run=/home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1
mkdir -p "$run"

python -u -m baselines.B5_control_pcc.prepare \
  --old-run /home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1 \
  --control100-run /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1 \
  --reference-report /home/qht/huawei/results/B5-rescore-reliable-I/default_20261017_192455/correlation_score.json \
  --epochs 20 \
  --output "$run/shared" \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

这是CPU校验阶段，显卡不动是正常的。大缓存的SHA检查会花时间，终端会打印校验进度。
成功后`ready=true`、`train_records=156`、`cache_regenerated=false`、`optimizer_updates=0`。
只写新study、原样复制初始化权重及很小的`cache_view.json`，不复制或重写任何npz，不执行历史代码加载器。
新study记录旧代码/数据来源，但使用本分支自己的新代码指纹，不能强改旧study的SHA绕过保护。

验证缓存仍存40条目标、92条配对，是存储范围；新评分排除坏I，预计：

|组|保留配对数|保留唯一目标数|
|---|---:|---:|
|Task1|27|21|
|Task2机器|39|39|
|Task2体脂秤|24|24|

对应两个Task2配对都指向d12_f2efb4a4c95a40。可靠I过滤在每次验证和选best之前执行，不是训练完后才重算。
预检会按缓存实际mask打印各组保留数量，不通过修改文件删掉该记录。
两组第0轮均应复现新参考：r1=0.70235417、r2=0.69846904、平均r=0.70041161（容差2e-5）。不一致会在首轮更新之前停止。

prepare成功后不要再次运行。`FileExistsError`表示新shared已经存在，先查看preflight.json再启动训练。
若中途失败，保留日志，用新的运行名如partner_v2诊断重试，不能覆盖旧study。

## 4. 两个终端并行训练，GPU0与GPU1

终端A：对照，训练PCC使用原FM快残差。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1 \
  --group control_old_fm \
  --gpu 0
```

终端B：实验，训练PCC使用control100快残差。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1 \
  --group adapt_control100 \
  --gpu 1
```

这是两个独立终端。放在同一终端、不加后台符号时会按顺序运行。
每个脚本都会依次完成：第0轮验证 → 慢CNN继续训练20轮（每5轮验证）→ best/last完整评估 → best便携模型导出。
第三张卡暂不需要启动其他改动，先隔离这个变量。网络、损失、感受野和β都不改。
训练时只更新慢CNN；验证阶段FM快残差直接读缓存，仍对应Heun16/K16。
日志放在输出目录旁，避免tee创建run.log导致“输出目录非空”。

## 5. 中断续跑

不重新prepare、不重新生成缓存。以实验组为例：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc

bash scripts/run_b5_control100_pcc.sh \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1 \
  --group adapt_control100 \
  --gpu 1 \
  --resume /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1/adapt_control100/last.pt
```

对照组将group、resume目录改为control_old_fm，并选择GPU0。只允许本实验同组、同目录、同study的精确续跑。
旧SlowPCC的last不能传给这里的resume；旧PCC20已经作为新实验共同初始化，不会丢掉其权重。

## 6. 看结果与打包

```bash
run=/home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1

cat "$run/control_old_fm/validation_metrics.csv"
cat "$run/adapt_control100/validation_metrics.csv"
```

两组都结束后，自动汇总best/last和相对第0轮的变化：

```bash
cd /home/qht/huawei_B5_control100_pcc
python -m baselines.B5_control_pcc.compare \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1
```

结果在`partner_v1/comparison.csv`和`comparison.json`。

先确认两组第0轮相同。然后比较同轮、各自best和last的完整记录R，以及每个导联/患者的变化。
best可能仍是第0轮，这表示新训练未超过起点，不是误读了旧权重。
若adapt相对自身第0轮和对照都改善，才支持新搭档适配有效；两组都提高则还包含继续训练的收益。
微小提高还需后续配对seed复核，不能保证测试集一定提高。慢r只辅助诊断，最终选择仍看完整ECG的两任务平均r。

两组结束后打包报告，不含pt、npz、npy：

```bash
cd /home/qht/huawei/results/B5-best-exp-Control100PCC

tar --exclude='*.pt' --exclude='*.npz' --exclude='*.npy' \
  -czf /home/qht/huawei/B5_control100_pcc_reports.tar.gz \
  partner_v1
```

下载`/home/qht/huawei/B5_control100_pcc_reports.tar.gz`即可。
模型导出保持原B5-best推理格式，但明确标记reference_score_protocol为新可靠I范围；不把新参考误当成旧范围。
