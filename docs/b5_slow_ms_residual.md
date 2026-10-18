# B5慢CNN多尺度残差修正：服务器实验操作

## 1. 本次只验证这一项新策略

分支：`B5-best-exp-SlowMSResidual`。

本地工作树：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-SlowMSResidual`。

服务器代码工作树：`/home/qht/huawei_B5_slow_ms_residual`。

结果目录：`/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1`。

|项目|control|residual|
|---|---|---|
|共同起点|旧seed42 ALL20最佳慢CNN|同一份最佳慢CNN|
|原慢CNN|继续更新全部参数|冻结，始终eval，无梯度|
|新模块|无|三路多尺度卷积修正，输出层零初始化|
|慢预测|原CNN输出|原CNN输出s0＋新分支Δs|
|训练损失|原SlowPCC损失|完全相同|
|新增分尺度辅助损失|无|无|
|FM|同一control100，冻结|相同|
|输入|50秒I＋人口学|相同；新分支还可读取原CNN自己的预测|
|训练|seed43，2e-5，20轮，record batch=2|相同|
|推理与组合|Heun16/K16，seed42，β=0.75|相同|
|选模|完整120秒的R=(r1+r2)/2，包含第0轮|相同|

这是“全量微调”与“保留原网络、训练新修正模块”两套策略的比较。它同时改变结构和可训练参数范围，若有效，再拆分归因；不能把收益单独归因于某一条卷积路径。

原损失保留：慢标签Huber＋0.1×慢PCC＋原F残差项＋0.1×最终组合PCC。没有均值监督，没有额外分尺度辅助项。起点虽然已经用ALL训练过，但本轮两组都使用上述同一原损失。

新模块24通道，三路各4个残差卷积块，膨胀率分别为[1,2,4,8]、[4,8,16,32]、[16,32,64,128]；通道LayerNorm不在时间轴去均值，人口学FiLM保留。单看新分支卷积路径，名义感受野约2.5/9.7/38.5秒；由于还读取原CNN的预测，最终依赖范围并不严格局限于这些数字。没有新增Transformer。

control可训练参数213,511；residual只训练新增93,031个参数，原213,511个参数冻结。

网络输出仍是当前50秒上下文上的预测，裁取中央10秒，拼接120秒。慢CNN及新分支都不读取真实目标、目标均值或FM缓存。FM缓存只在训练损失和固定搭档评估时使用。F监督不等于严格低通滤波，不能保证新分支完全没有快速起伏，需要看最终组合r和波形。

## 2. 默认初始化位置：本次用ALL20，不用旧PCC20

```text
/home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1/
    shared/study.json
    shared/cache_view.json
    shared/components.pt
    adapt_multiscale/best.pt
```

该best应为旧seed42的第20轮，可靠I评分约：Task1=0.705619，Task2=0.697503，平均R=0.701561。

读取既有control100缓存：

```text
/home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache/
```

prepare会校验旧实验及缓存指纹，然后把选定best的慢CNN权重装入本次初始化。不会错用shared/components.pt里旧的PCC20慢权重。FM保持同一control100。

无需重新生成FM缓存，无需修改原B5或旧实验的任何文件。新study保存独立组件包和只读缓存索引，用于两组复用、续跑及校验。

## 3. 本地推送，服务器新建工作树

本地代码提交完成后，由你推送并同步到Gitee：

```powershell
cd "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-SlowMSResidual"
git push origin B5-best-exp-SlowMSResidual
```

以下为服务器操作。Gitee已有该分支后执行一次：

```bash
conda activate BioFlow
cd /home/qht/huawei

git fetch gitee B5-best-exp-SlowMSResidual

git worktree add --detach \
  /home/qht/huawei_B5_slow_ms_residual \
  gitee/B5-best-exp-SlowMSResidual

cd /home/qht/huawei_B5_slow_ms_residual
git log -1 --oneline
```

若目录已存在，不重复worktree add；保留已有实验运行代码，先确认是否就是本分支。训练开始后，不在该工作树切换代码；study会校验代码指纹。

## 4. 准备两组共同起点：只运行一次，不训练

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_slow_ms_residual

run=/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1
mkdir -p "$run"

python -u -m baselines.B5_slow_ms.prepare \
  --source-run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --control100-run /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1 \
  --training-seed 43 \
  --learning-rate 2e-5 \
  --epochs 20 \
  --output "$run/shared" \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

成功应看到：ready=true、source_best_epoch=20、zero_init_max_abs_difference=0、optimizer_updates=0、FM_optimizer_updates=0、extra_scale_loss_weight=0。reference_score约为0.701561，而非0.700412。CPU前向检查不是训练。

训练记录应为156，Task1保留27配对/21目标，Task2机器39、体脂秤24目标。标差I整条不评分，标差目标导联单独屏蔽。

首次扫描缓存需要读取文件和计算SHA，GPU不动是正常的。成功后直接启动下一步；不要再次运行prepare到同一个shared目录。若确实要新实验，换整个run目录，不能清空已有结果来复用名字。

## 5. 两张卡分别运行，可并行

每组自动完成：第0轮正式验证 → 20轮训练 → best/last独立评估 → best便携模型导出。第0轮验证必须复现共同起点分数，否则会停止，避免用错误权重或缓存训练。

在两个独立终端或tmux会话分别执行。脚本前台运行，不会自动后台；第一组不结束也不妨碍另一终端启动第二组。

终端A，GPU0，对照组：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_slow_ms_residual

bash scripts/run_b5_slow_ms_residual.sh \
  --run /home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1 \
  --group control \
  --gpu 0
```

终端B，GPU1，多尺度修正组：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_slow_ms_residual

bash scripts/run_b5_slow_ms_residual.sh \
  --run /home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1 \
  --group residual \
  --gpu 1
```

GPU号可以替换为当前空闲卡，程序内部仍用cuda。每组应20轮、1560次优化器更新（156记录、batch2时）。实验组每轮检查原慢CNN和FM的权重摘要未变化。

## 6. 查看进度与两组结果

```bash
run=/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1

tail -n 5 "$run/control.train.log"
tail -n 5 "$run/residual.train.log"
```

两组全部结束后：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_slow_ms_residual
run=/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1

python -m baselines.B5_slow_ms.summarize --run "$run"
cat "$run/summary.csv"
)
```

首先看是否超过初始化0.701561，再看residual与control差多少。best仍为0轮意味着没有得到新最佳，不能把相同best分数理解为没启动训练；此时看last和validation_metrics.csv。

不按单个任务、单条患者或最有利随机种子挑结果。若有收益，再做另一个共同训练seed复核；现在不默认启动更多组。

## 7. 中断后续跑

例如residual中断，保持代码、study、cache、输出路径、GPU可见卡数一致，再执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_slow_ms_residual

bash scripts/run_b5_slow_ms_residual.sh \
  --run /home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1 \
  --group residual \
  --gpu 1 \
  --resume /home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1/residual/last.pt
```

control同理，group改control，resume指向control/last.pt。必须保留同组best.pt，避免丢失第0轮候选。只从已保存的完整epoch恢复，不恢复中断epoch中的半轮更新。

## 8. 打包给我分析

两组评估、导出、summary全部成功后：

```bash
tar -czf /home/qht/huawei/B5_slow_ms_residual_reports.tar.gz \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-best-exp-SlowMSResidual \
  ms_residual_v1
```

下载：`/home/qht/huawei/B5_slow_ms_residual_reports.tar.gz`。

本轮保留了每次验证预测数组，若需要进一步看波形再定向打包，不在首次报告包中传输大权重或数组。

## 9. 便携模型与测试

两组best包的明确路径指针分别在：

```text
/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1/control/latest_best_bundle.txt
/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1/residual/latest_best_bundle.txt
```

新分支兼容`baselines.B5_best.predict --checkpoint 便携包.pt`及`baselines.B5_best.validate --checkpoint 便携包.pt`。必须在本分支代码上加载新残差模型。原CNN便携包的加载行为保持不变。

真正预测新患者时只需完整120秒500Hz原始μV的I和人口学，输出12导联，其中I保留输入；其余11导联由FM与慢预测组合。模型包包含全部权重和尺度，不依赖训练缓存或真实目标。

周期评估和best/last评估复用历史每配对FM预测，保证本轮比较只受到慢预测变化影响。新患者预测会从噪声重新运行FM；记录key影响噪声，不能把不同key的新采样与历史缓存逐点相同当作必要条件。

## 10. 本地功能验证范围

本地只运行合成CPU前向/反向、数据/QC评分、指纹保护、检查点与便携模型加载检查。不执行optimizer.step，不启动华为训练，不声称有性能提升。服务器第0轮真实分数复现是开始训练前的最后一道功能核对。

本次实现已通过13项新旧CPU功能检查，包括：最佳权重与旧组件初始化不同仍能正确加载、零初始化精确复现、冻结参数无梯度、非零修正模型独立推理、坏I与坏目标导联过滤、源缓存不被改写、篡改缓存拒绝加载、便携模型往返加载，以及原尺度实验回归。运行脚本和本文8段Bash命令均通过语法检查。
