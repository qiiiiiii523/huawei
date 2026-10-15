# B5-S50：Control / S50 / S50-F 三组FM条件接入

本分支使用同一个FM初始化，比较不接慢条件、接S50、接S50-F。F已经用于慢CNN的预训练，接入后慢CNN冻结，**三组FM损失相同，不把F损失加进FM**。本地仅完成功能检查，不启动真实训练。

## 1. 运行组和评分

|组名|FM|条件来源|目的|
|---|---|---|---|
|control|原B5继续微调|无慢条件|同预算对照|
|slow|原B5＋瓶颈零初始化残差|冻结S50原损失CNN|慢条件是否有效|
|slow_f|与slow相同|冻结S50-F CNN|更换慢预测器是否进一步有效|

后两组的条件编码器结构和初始参数相同，默认64/128/256主干时新增9536参数。原同步I、人口学、生成时间t、FM噪声、解码器和I输出头保持原作用。固定SlowCNN不进入优化器。

三组均从原FM EMA初始化，新优化器，新增20轮，每5轮验证。学习率、batch、累计梯度、EMA、原损失、训练质量mask等继承原FM检查点；采样固定Heun16/K16/seed42，拼接120秒。

**QC v3总分为唯一best.pt选择依据**；best_r.pt按QC主相关性，best_bonus.pt按QC加分，last.pt用于续跑。第0轮也可入选。每次验证和最终评估对同一批预测另算完整集合v2，存到full_population/，只作历史参考，不据此重新选模型。两套评分不额外训练或重复生成。

S50与S50-F各自看50秒I、输出中间10秒；先预测并拼好120秒慢曲线，再裁取FM窗口。端点使用重复边界，全记录坐标插值回500Hz；不逐窗去均值、不额外放大、不使用真实慢曲线或均值作为条件。

## 2. 本次固定的输入文件

```text
FM共同初始化（不是best_r.pt）：
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt

S50：
/home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints/best_score.pt

S50-F：
/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/best_score.pt

原慢实验prepared：
/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
```

FM初始化仍按你已指定的原best.pt，不自动替换QC重评推荐的其他候选。它必须与prepared绑定的SHA一致，保证第0轮参考可复现；如果决定换FM初始化，需要另做经过核验的参考准备，不能直接改成best_r.pt。

S50/S50-F必须来自同一S30初始化、同一prepared、同一seed及20轮预算。程序验证这些元数据，并分别核对C/F身份。各慢权重旁原evaluation_best_score存在时，会把固定输出组合也按当前QC与完整集合重算，用于比较；它仍是组合参考，不是新FM训练结果。缺少原组合评估时仍可训练，但比较表不会虚构这两项参考。

## 3. 推送、更新服务器独立worktree

本机PowerShell：

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\FM-S50'
git push origin B5-S50
```

服务器远程名按实际配置使用，下面用gitee；该远程需包含本次提交。本地origin与服务器gitee若为不同仓库，先同步。

首次建立代码worktree：

```bash
cd /home/qht/huawei
git fetch gitee B5-S50
git worktree add --detach /home/qht/huawei_fm_S50 FETCH_HEAD
```

如果该worktree已存在，使用下面这段更新，不再重复创建：

```bash
cd /home/qht/huawei_fm_S50
git fetch gitee B5-S50
git switch --detach FETCH_HEAD
```

随后：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
sed -i 's/\r$//' scripts/run_b5_s50_pair.sh
bash -n scripts/run_b5_s50_pair.sh
```

脚本文件名沿用pair，但现在支持三个组。代码更新只在此worktree进行，不切换原huawei主目录或正在运行QC重评的worktree。

## 4. 一次性准备两种条件及QC参考

**这是新协议和新输出目录。旧fm_condition_v1/shared及旧B5-S50-v1续训状态不能复用或覆盖。** 原S50/S50-F权重及原prepared仍可以复用，不用重训慢CNN。

```bash
cd /home/qht/huawei_fm_S50
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
root=/home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2
mkdir -p "$root"
set -o pipefail

python -u -m baselines.B5_s50.prepare \
  --fm-checkpoint /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt \
  --slow-checkpoint /home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints/best_score.pt \
  --slow-f-checkpoint /home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/best_score.pt \
  --prepared /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared \
  --output "$root/shared" \
  --device cuda \
  2>&1 | tee "$root/prepare.log"

python -u -m baselines.B5_s50.check \
  --shared "$root/shared" \
  --report "$root/preflight.json" \
  2>&1 | tee "$root/preflight.log"
```

准备只做核验与冻结CNN推理，没有优化器更新。先核验大文件SHA、原始目标逐点一致性、输入I/人口学、两慢CNN配对；读盘期间GPU暂时没动正常。缓存分别为train/validation_slow.npy和train/validation_slow_f.npy。

预检期望ready=true、optimizer_updates=0、experimental_adapter_same_initialization=true；groups.slow和groups.slow_f的速度场/ODE差值都为0，投影梯度都大于0。它是功能检查，不是ECG效果。正式运行还会核对三组第0轮实际K16的QC分和完整集合分与共同原B5参考一致。

成功后不要重复prepare；若中途失败保留原目录，排查后使用一个新的root重新准备，并让后面三组都传同一个 --root。

## 5. 三个终端分别训练与自动评估

先等现有QC重评或旧训练结束，确认相应GPU可用。每组一张卡；三个终端可并行，同一终端粘贴会顺序执行。

终端A，GPU0，对照：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh --group control --gpu 0
```

终端B，GPU1，接S50：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh --group slow --gpu 1
```

终端C，GPU2，接S50-F：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh --group slow_f --gpu 2
```

每组新增20轮，先完成第0轮完整验证再开始训练。K16验证计算较多，不能把验证耗时当作卡住。训练后自动评估best、best_r、best_bonus、last；同一组相同EMA的候选可以复用生成，但不同慢来源/组别/评分协议不复用。训练和日志位于root/control、root/slow、root/slow_f。

中断后在**原组命令末尾加 --resume**，恢复该组checkpoints/last.pt。不能用它续跑旧B5或旧版B5-S50。脚本不是后台服务，可在已有tmux会话中运行。

脚本会核对已完成评估与当前权重的SHA；不一致或中断的评估目录自动改名保留，然后重新评估，不再静默跳过旧结果。

## 6. 三组都完成后汇总

```bash
cd /home/qht/huawei_fm_S50
root=/home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2
python -m baselines.B5_s50.compare \
  --shared "$root/shared" \
  --control "$root/control" \
  --slow "$root/slow" \
  --slow-f "$root/slow_f" \
  --output "$root/comparison"
```

查看comparison/report.md、scores.csv及deltas.json：

- slow vs control：慢条件是否有效。
- slow_f vs control：S50-F条件是否有效。
- slow_f vs slow：更换慢预测器是否进一步改善。
- 各组 vs原B5、固定输出组合：是否值得替换现有方案。

主要结论用QC同口径比较；完整集合分另列。检查各组evaluations/best的wave_summary.csv、wave_record_lead.csv及plots，关注正常导联慢/快r、独有患者、RMSE、波峰和10秒接缝。best若仍为0轮，就不能声称该组继续训练已取得收益。单seed有收益后再复核，不预设S50-F一定胜过S50。

打包轻量报告：

```bash
cd /home/qht/huawei
find results/B5-S50/fm_slow_qc_three_v2 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o \
     -name '*.md' -o -name '*.log' -o -name '*.png' \) \
  -print0 | tar --null -T - -czf B5_S50_three_QC_reports.tar.gz
```

## 7. 两种慢条件均支持新记录推理

训练产生的检查点内嵌其对应的冻结CNN，推理自动选择该权重，不需要prepared、原慢CNN路径或真实目标。

```bash
cd /home/qht/huawei_fm_S50
CUDA_VISIBLE_DEVICES=2 python -m baselines.B5_s50.predict \
  --checkpoint /home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2/slow_f/checkpoints/best.pt \
  --anchor-npy /path/to/new_I_uV.npy \
  --metadata-json /path/to/new_metadata.json \
  --record-key new_record_001 \
  --output /home/qht/huawei/results/B5-S50/new_record_001 \
  --device cuda
```

输入为500Hz同步I的[60000]或[1,60000]原始μV数组；人口学JSON可包含age、sex/gender、height、weight，允许缺失。默认Heun16/K16，输出[12,60000]原始μV。把checkpoint换成slow组即可使用原S50条件。该方案需要整条120秒输入及前后I，不是实时因果推理；测试不使用QC标签修正输出，QC仅用于本地评分。

## 8. 本地验证范围

30项CPU合成数据及既有B5回归检查通过，Bash语法和Git差异格式检查通过。包含两种CNN来源/配对校验、条件缓存区别、零初始化输出及投影梯度、同一适配器初始化、QC与完整集合评分分离、三组第0轮核验、候选保存/恢复、禁止跨组复用、三组汇总和仅I/人口学的新记录推理。

没有运行真实ECG训练，没有执行优化器更新。文件流程检查中的生成使用合成替身，真实ODE等价性用短合成信号单独检查；不能把测试夹具中的分数或20轮完成标记当作实验结果。服务器真实权重、文件、GPU显存及性能仍需按上述预检和实验验证。

## 9. 已完成20轮后，再续训20轮（总预算40轮）

本节直接使用已有的三个组目录及原shared，不重新prepare、不修改shared.json、不重新预训练慢CNN。只继续control与slow也可以；slow_f为可选第三组。

新版代码显式兼容提交7738de6生成的原缓存；兼容检查同时核对原实现及本次已审核实现的完整哈希集合。其他代码或数据发生变化仍拒绝续训，不是关闭检查。

先在本机将本次提交推送到服务器能获取的远程，再在服务器更新独立worktree（没有训练进程运行时执行）：

```bash
cd /home/qht/huawei_fm_S50
git fetch gitee B5-S50
git switch --detach FETCH_HEAD
conda activate BioFlow
sed -i 's/\r$//' scripts/run_b5_s50_pair.sh
bash -n scripts/run_b5_s50_pair.sh
```

终端A，GPU0，对照组从自己的last.pt继续：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh \
  --group control --gpu 0 \
  --resume --total-epochs 40
```

终端B，GPU1，S50组从自己的last.pt继续：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh \
  --group slow --gpu 1 \
  --resume --total-epochs 40
```

可选终端C，GPU2，S50-F同预算续训：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh \
  --group slow_f --gpu 2 \
  --resume --total-epochs 40
```

`40`表示从原B5初始化起累计新增40轮，即此次继续21～40轮，不是额外再加40轮。恢复raw模型、EMA、AdamW状态、更新次数、随机数和DataLoader随机状态；学习率、损失、Heun16/K16、QC选模规则均不改变，best候选保留此前纪录。

首次延长前，自动把各组checkpoints顶层的权重、历史、配置及完成记录复制到checkpoints_epoch20，并保存校验清单；不复制庞大的validation预测数组。前20轮evaluations及comparison保留。新评估位于evaluations_epoch40，训练历史在原history.jsonl中追加。

启动日志应显示start_epoch=20、total_epochs=40、remaining_epochs=20。中断后重复原续训命令，恢复最近完成的轮次，不额外重置优化器。若已完成40轮，重复命令不会变成60轮。只能在一个已完成预算边界显式延长，不能在尚未完成20轮时改预算。

两组全部训练和自动评估完成后汇总（不要求slow_f也训练到40）：

```bash
cd /home/qht/huawei_fm_S50
root=/home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2
python -m baselines.B5_s50.compare \
  --shared "$root/shared" \
  --control "$root/control" \
  --slow "$root/slow" \
  --total-epochs 40 \
  --output "$root/comparison_epoch40"
```

若三组都续训到40轮，使用下面这条替代上面的两组汇总：

```bash
cd /home/qht/huawei_fm_S50
root=/home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2
python -m baselines.B5_s50.compare \
  --shared "$root/shared" \
  --control "$root/control" \
  --slow "$root/slow" \
  --slow-f "$root/slow_f" \
  --total-epochs 40 \
  --output "$root/comparison_epoch40_three"
```

程序检查参与汇总的组均已完成相同预算，以及当前权重SHA与对应预算评估一致，不允许混用20轮和40轮。best若留在第20轮甚至第0轮仍是合法结果；last必须到第40轮。第一次和第二次比较目录不同，不能把旧20轮评估复制过去充作新评估。

本次续训变更在本地通过5项不依赖PyTorch的测试：预算延长/中断恢复规则、审核过的旧实现兼容、归档与校验、评估目录及权重匹配、两组/三组同预算汇总与混合预算拒绝；Python解析及Bash语法也通过。本机当前没有可用PyTorch，新增的优化器/EMA/RNG恢复集成测试未执行，没有启动训练。服务器可在启动续训前运行以下合成检查（不执行优化器更新）：

```bash
cd /home/qht/huawei_fm_S50
python -m unittest \
  baselines.B5_s50.tests.test_continuation \
  baselines.B5_s50.tests.test_integration -v
```
