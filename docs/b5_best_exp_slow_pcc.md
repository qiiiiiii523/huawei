# B5-best-exp-SlowPCC：冻结FM，只优化慢CNN的最终组合相关性

## 1．为什么做这一轮

当前B5-best为：

```text
原FM输出 p → p−L(p) ──────────────────┐
                                     ├→ 最终120秒ECG
同步I＋人口学 → S50-F → 0.75×预测慢曲线 s ┘
```

`L`仍为完整120秒上的501点居中移动平均；输出为`p−L(p)+0.75×s`。两组均固定当前B5-best的β=0.75。

S50-F原训练目标是接近真实慢曲线，但最终评分取决于它与FM快残差相加后的波形。已有β探针显示：同一慢预测的单独r不变，改变相对贡献却能改变最终r。这支持测试组合监督，但不保证有效。

本轮从最新`B5-best/7899e68`创建独立分支`B5-best-exp-SlowPCC`，包含新的β0.75及旧包兼容实现。不合并V13模块，不改FM路径、FM网络、人口学、输入范围或采样，不加入M均值监督。

## 2．对照与实验，只差一项

|项目|control|pcc|
|---|---|---|
|慢CNN初始化|当前同一个S50-F best_score.pt|相同|
|FM|原B5权重固定，只使用共享预测缓存|完全相同|
|可训练网络|S50-F|S50-F|
|保留的原损失|慢Huber＋0.1×慢PCC＋原固定F监督|相同|
|额外损失|0|0.1×最终组合120秒PCC损失|
|输入|当前50秒同步I上下文＋人口学|相同|
|预算|新增20轮|相同|
|学习率、batch|读取原S50-F配置，通常1e-4、2条记录|相同|
|优化器|新建AdamW，不继承旧优化器|相同|
|验证|第0轮、每5轮及最后一轮|相同|
|选模|最终组合(R1+R2)/2，明确导联QC|相同|
|FM采样|Heun16/K16/seed42|相同|
|最终组合系数|β=0.75|相同|

F系数从原权重配置读取，预期为`2.5890171538829843`；宽度仍为50Hz下51点。没有重新标定F权重，没有逐窗口去均值。

新增PCC先把预测慢曲线按原block中心坐标插值回500Hz，与固定FM快残差相加，再对完整60000点计算。训练批次对有效记录/导联平均；正式验证继续保留目标记录等权、Task2两设备等权，不把训练损失当作两个任务的正式汇总分。

真实波形只用于训练监督和验证评分。慢CNN输入接口严格只接收I和人口学；不会接收真实慢曲线、目标均值或真实峰位置。

## 3．目录与原权重

本地worktree：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-SlowPCC`。

服务器新代码worktree：`/home/qht/huawei_B5_best_exp_slow_pcc`。

数据、原权重和新结果根目录仍为`/home/qht/huawei`。预检按`configs/b5_best.json`读取：

```text
固定FM：
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt

慢CNN初始化：
/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/best_score.pt

原数据说明：
/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared/prepared.json
```

这里不使用B5-S50慢条件注入实验的FM权重，也不使用V13实验权重。

## 4．推送后，服务器创建独立worktree

本地先推送新分支。服务器BioFlow环境首次执行：

```bash
(
set -e
cd /home/qht/huawei
git fetch gitee B5-best-exp-SlowPCC

git worktree add --detach \
  /home/qht/huawei_B5_best_exp_slow_pcc \
  gitee/B5-best-exp-SlowPCC

cd /home/qht/huawei_B5_best_exp_slow_pcc
git log -1 --oneline
)
```

如果目录已经存在，在该worktree中更新：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc
git fetch gitee B5-best-exp-SlowPCC
git switch --detach FETCH_HEAD
```

不要在训练中更新实现。study、缓存及续训都绑定本轮代码和数据指纹。

## 5．CPU预检：不训练

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_slow_pcc

run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1
mkdir -p "$run"

python -u -m baselines.B5_slow_pcc.prepare \
  --repository-root /home/qht/huawei \
  --output "$run/shared" \
  --epochs 20 \
  --combo-weight 0.1 \
  2>&1 | tee "$run/prepare.log"

cat "$run/shared/preflight.json"
)
```

成功应为`ready:true`、`training_started:false`，新旧慢预测差为0，插值差接近0；同时列出源学习率、batch、F系数和新增PCC系数。

预检检查源权重、训练/验证数据、人口学、QC、患者划分及旧FM验证预测。`shared/components.pt`只是本轮固定组件副本，原权重不会被修改。

预检成功后不用再次运行；重复prepare会拒绝非空shared。要改预算/系数，使用新的run目录，两个训练组都用同一份新study。

## 6．GPU生成一次训练集FM缓存

这一阶段使用GPU1。它是**固定FM推理，不是训练**，没有优化器更新；Heun16/K16计算量较大，应等缓存全部完成后再开两组训练。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_slow_pcc

run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_slow_pcc.cache \
  --study "$run/shared/study.json" \
  --output "$run/cache" \
  --device cuda \
  --execute-inference \
  2>&1 | tee "$run/cache.log"
)
```

预计处理原156条训练目标，各自按12个窗口从噪声生成；验证40条目标复用原FM的Heun16/K16预测，仅用于验证。两部分单独记录、患者不重叠。

缓存保存原始目标、I/人口学、质量mask、完整记录慢标签和固定FM输出，通常需要约1–2GB磁盘。训练时不再反复生成FM，每轮只训练慢CNN。

完成标志为`cache/cache.json`中的`completed:true`。每条记录完成后保存进度。若这一阶段中断，用相同路径加`--resume`：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_slow_pcc.cache \
  --study "$run/shared/study.json" \
  --output "$run/cache" \
  --device cuda \
  --resume \
  --execute-inference \
  2>&1 | tee -a "$run/cache.log"
)
```

已验证完成的记录会跳过。不要把验证预测拷进train子目录，也不要更换FM采样seed或K。

## 7．两个终端并行训练，分别GPU0和GPU2

先确认缓存完成。终端A：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

bash scripts/run_b5_slow_pcc.sh \
  --study "$run/shared/study.json" \
  --cache "$run/cache" \
  --group control \
  --output "$run/control" \
  --gpu 0
```

终端B：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

bash scripts/run_b5_slow_pcc.sh \
  --study "$run/shared/study.json" \
  --cache "$run/cache" \
  --group pcc \
  --output "$run/pcc" \
  --gpu 2
```

脚本自动完成：第0轮初始验证→新增20轮慢CNN训练→best/last正式评估→导出最佳推理包。第0轮应复现当前β0.75 B5-best约0.69492的QC平均r，超出容差会停止训练并保留报告。

本轮两组都固定β=0.75，训练中的组合PCC、周期验证及导出推理也统一使用0.75，不再把比例改变当作实验因素，不叠加V13。第0轮也参与选模；若续训全部变差，best会保留第0轮。只有`best.pt`和`last.pt`两种训练权重，没有RMSE/bonus选模。

此前V13若仍按β=1运行，不能把其绝对分数与本轮β=0.75直接当作模块效果比较；先分别看各自同设置对照的收益。

日志在`control.train.log`、`pcc.train.log`，放在对应结果目录旁边。不要把tee日志放进首次要求为空的输出目录。

## 8．训练中断后续训

例如pcc组，继续原20轮总预算：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

bash scripts/run_b5_slow_pcc.sh \
  --study "$run/shared/study.json" \
  --cache "$run/cache" \
  --group pcc \
  --output "$run/pcc" \
  --resume "$run/pcc/last.pt" \
  --gpu 2
```

对照改为control/GPU0。保持原study、cache、group、输出目录和实现。续训恢复慢CNN优化器、随机状态和加载顺序，不是另加20轮。

## 9．两组结束后生成对比，不用GPU

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1
report="$run/comparison_$(date +%Y%m%d_%H%M%S)"

python -u -m baselines.B5_slow_pcc.compare \
  --run-root "$run" \
  --output "$report" \
  2>&1 | tee "${report}.log"

cat "$report/report.md"
)
```

|文件|检查内容|
|`summary.csv`、`report.md`|两组best/last，两个任务及平均r、选中轮次|
|`lead_comparison.csv`|全部11导联，Task2两设备，是否有其他导联下降|
|`morphology_summary.csv`、逐记录表|V1/V3/V5/V6的峰谷、时间、峰宽和局部形态|
|PNG|相同片段原始波形、完整记录慢曲线，检查波峰细节是否损害|
|每次验证的`experiment.json`|慢r、慢RMSE诊断以及最终组合分数|
|`history.jsonl`|原Huber、慢PCC、F项、新增组合PCC及其加权项|

主要判断：实验best能否超过同预算对照best；慢曲线和最终r是否一致改善，快形态有无变差。不能只因新增PCC损失下降就宣布有效。

只有明确导联QC正常的目标参与正式评分；局部心搏诊断另跳过坏I。图片和局部r不是正式分数，不会按真值平移、缩放预测。RMSE只保留诊断作用。

## 10．推理与后续使用

每组训练完成后，脚本导出`best_bundle_时间.pt`：固定原FM＋该组best慢CNN。可直接使用原`baselines.B5_best.predict`接口，默认仍Heun16/K16/seed42，β=0.75，系数保存在包内。

这个包不需要训练缓存、不需要真实目标，只输入新患者120秒I和可用人口学信息。

`best.pt`/`last.pt`是实验训练状态；部署用导出的`best_bundle_时间.pt`。缓存和训练状态不是测试输入。导出不自动将新模型替换为稳定B5-best，是否采用要先看配对结果。

## 11．打包给分析

```bash
cd /home/qht/huawei

tar \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -czf B5_SlowPCC_reports.tar.gz \
  results/B5-best-exp-SlowPCC/combo_pcc_v1
```

会包含预检、缓存说明/进度、训练日志、曲线、best/last评分、全部导联和局部形态对比、图片；不包含大型数组和权重。

本轮没有在本地训练，没有声称新增监督已经提高r。
