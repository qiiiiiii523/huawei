# 体脂秤context：O / E / E0三组历史编码实验

本次只改变历史特征编码。FM、慢CNN、d6、评分、当前I和原预测II的编码方式不变，不加入门控。旧编码和旧权重仍可原样使用，不会自动替换默认模型。

## 1. 三组各验证什么

|项目|O：原编码|E：形态＋幅度编码|E0：新编码不截断|
|---|---|---|---|
|历史I/II模板归一化|两条都除以I的RMS|分别除以各自RMS|与E完全一致|
|历史心搏模板波形截断|±8|±8|不截断|
|27维历史关系统计|共享I尺度，截断后计算|截断前相关性21项＋相对I的log幅度比6项|与E逐值相同|
|模板尺寸、模型参数|2×300，原BodyIIAdapter|相同|相同|
|初始权重、种子、预算|相同随机初始化，seed42，10轮|相同|相同|
|损失与选模|10秒PCC＋0.01修正量平方；完整120秒QC正常II的目标记录等权r选best|相同|相同|

- E对O：检验新编码这一组改动，包括分别归一化、保留截断前关系统计、分开表达幅度。不能把收益全部归因于其中某一项。
- E0对E：只检验历史心搏模板的±8截断是否有必要。当前I、原预测II的波形保护，以及log幅度比的数值范围完全不变。

E/E0使用相同的历史I事件检测、相同的最多32个心搏、相同的逐点中位数。历史与当前目标属于不同时间，不进行对齐。相关性与幅度统计在同一批有效历史片段上提取，模板和统计只使用测试可见的历史。

E0保留非有限值检查、原质量mask、1e-4尺度下限。它允许正常有限的归一化尖峰超过8，不会为了幅度大而自动删除记录。训练中保留原有的非有限loss检查和梯度裁剪，不保证E0一定更好。分别归一化之后也可能没有任何值超过8，此时E0与E接近相同是有效实验结果。

本轮小修正器从相同随机权重训练，**不是从C-best继续训练**。延长C训练属于另一个实验。三组都可在新运行开始前统一指定`--epochs 20`，不能只增加一组预算。

## 2. 现有缓存和输出目录

服务器代码工作树默认`/home/qht/huawei_B5_best_context`，不同目录名请修改cd。原缓存只读复用：

```text
/home/qht/huawei/results/B5-best-context/body_v1/cache
```

三组脚本使用新目录，避免覆盖之前O/E或C/G的结果：

```text
/home/qht/huawei/results/B5-best-context/body_encoding_three_v1/
    run.json
    encoded_cache/       # E历史特征
    unclipped_cache/     # E0历史特征
    preflight_O*/
    preflight_E*/
    preflight_E0*/
    preflight_pair.json  # 三组同初始权重、同数据、同预算检查
    O/best.pt
    E/best.pt
    E0/best.pt
    study/               # 20组评估，实际目录见study_path.txt
```

缓存准备不运行B5，只重读原始历史、复现旧特征并生成新特征。三组I、基础预测、真实II、QC、可用性和划分逐数组保持一致。E/E0的27维统计也必须完全一致，否则报错。新权重自带编码标识，实时推理自动选择正确编码。不要把E/E0权重交给未更新的旧代码。

## 3. 更新服务器代码

本地保存提交后由你推送，服务器执行：

```bash
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
```

工作树若是detached HEAD也可快进。存在本地改动时先检查并保存，不使用强制重置。

## 4. 推荐：GPU1依次完成三组训练和评估

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
sed -i 's/\r$//' scripts/run_b5_body_encoding.sh
bash scripts/run_b5_body_encoding.sh --gpu 1
```

顺序：E/E0缓存准备 → 三组CPU前向、反向和修正上限检查 → 三组同初始化配对检查 → O训练 → E训练 → E0训练 → 20组评估。默认每组10轮、batch16、lr=0.0002，不重新生成FM预测，不连接网络、不安装依赖。日志在子目录外，避免tee造成“输出目录非空”。

中断后，保持相同设置续跑：

```bash
bash scripts/run_b5_body_encoding.sh --gpu 1 --resume
```

换训练种子验证时，B5推理噪声仍固定42，使用新目录：

```bash
bash scripts/run_b5_body_encoding.sh --gpu 1 --seed 43 \
  --run /home/qht/huawei/results/B5-best-context/body_encoding_three_seed43
```

**之前已经启动的O/E实验不需要作废。** 更新后续跑原两组可用：

```bash
bash scripts/run_b5_body_encoding.sh --gpu 1 --two-groups --resume
```

该命令默认继续`body_encoding_v1`旧目录。三组默认`body_encoding_three_v1`新目录。禁止在同一运行中途改变组别、预算或种子。若要复用已完成的O/E，只需用相同种子、预算另训E0，然后在评估入口同时指定原O/E目录和新的E0目录；入口会核对初始权重、配置、缓存来源。

## 5. 可选：GPU0 / GPU1 / GPU2同时训练

本节与上一节选择一种，避免对同一输出目录重复启动。

先在一个终端准备并完成三组CPU预检：

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1
source_cache=/home/qht/huawei/results/B5-best-context/body_v1/cache
config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
mkdir -p "$run"
python -u -m baselines.B5_best_context.body_encoding_prepare \
  --config "$config" --cache "$source_cache" --output "$run/encoded_cache"
python -u -m baselines.B5_best_context.body_encoding_prepare \
  --config "$config" --cache "$source_cache" --output "$run/unclipped_cache" \
  --encoding shape_amplitude_unclipped_v1

for group in O E E0; do
  selected_cache="$source_cache"
  if [ "$group" = E ]; then selected_cache="$run/encoded_cache"; fi
  if [ "$group" = E0 ]; then selected_cache="$run/unclipped_cache"; fi
  python -m baselines.B5_best_context.body_train \
    --cache "$selected_cache" --group context --seed 42 --epochs 10 \
    --output "$run/preflight_$group" --check-only || break
done
python -m baselines.B5_best_context.body_encoding_study \
  --config "$config" --cache "$source_cache" --encoded-cache "$run/encoded_cache" \
  --control-run "$run/preflight_O" --encoded-run "$run/preflight_E" \
  --unclipped-cache "$run/unclipped_cache" --unclipped-run "$run/preflight_E0" \
  --check-only --output "$run/preflight_pair.json"
```

最后一个命令出现`ready: true`后，分别在三个终端启动：

```bash
# 终端一：O，GPU0
cd /home/qht/huawei_B5_best_context
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5_best_context.body_train \
  --cache /home/qht/huawei/results/B5-best-context/body_v1/cache \
  --group context --seed 42 --epochs 10 --device cuda \
  --output /home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1/O
```

```bash
# 终端二：E，GPU1
cd /home/qht/huawei_B5_best_context
CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_train \
  --cache /home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1/encoded_cache \
  --group context --seed 42 --epochs 10 --device cuda \
  --output /home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1/E
```

```bash
# 终端三：E0，GPU2
cd /home/qht/huawei_B5_best_context
CUDA_VISIBLE_DEVICES=2 python -u -m baselines.B5_best_context.body_train \
  --cache /home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1/unclipped_cache \
  --group context --seed 42 --epochs 10 --device cuda \
  --output /home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1/E0
```

三组均生成completed.json后评估：

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_encoding_three_parallel_v1
CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_encoding_study \
  --config /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml \
  --cache /home/qht/huawei/results/B5-best-context/body_v1/cache \
  --encoded-cache "$run/encoded_cache" --control-run "$run/O" --encoded-run "$run/E" \
  --unclipped-cache "$run/unclipped_cache" --unclipped-run "$run/E0" \
  --output "$run/study" --device cuda
```

手动训练中断后在原命令加`--resume`，其余参数不变。评估输出必须为空，重算时用新目录名。

## 6. 看哪些结果

- `summary.csv`：E_both−O_both是新编码增量；E0_both−E_both是不截断历史模板的独立增量。检查Task1和机器组保持一致。
- O/E/E0各自的disabled和三个固定shuffled：判断是否利用个人历史，不挑有利的打乱结果。关闭context是推理消融，不等于重新训练的无context模型。
- `body_encoding_deltas.csv`：逐记录II r，E−O和E0−E，快/慢r差异。关注原先改善与下降记录能否同时改善。
- `body_component_metrics.csv`：II整体r、快r、慢r、RMSE和修正量。RMSE只作诊断，本次选模仍按r。
- 两个新缓存内的`clipping_comparison.csv`：`*_beat_exceeds_8_fraction`是归一化值超过8的比例；`*_beat_clipped_fraction`是实际截断比例，E0可前者非零、后者为0。
- `templates/*.csv`：固定患者的历史模板前后对比。E/E0尺度相同可直接比较，O尺度不同，不能当成同一μV纵轴比较。

20组评分复用同一基础预测，包括baseline、d6_only和每种编码的body/both/disabled/三组shuffled。完成标记是`study/encoding_checkpoints.json`。不预设不截断一定更好，也不预设±8必须保留；先看稳定性和最终r，再决定部署。

## 7. 打包报告

```bash
run=/home/qht/huawei/results/B5-best-context/body_encoding_three_v1
tar -czf /home/qht/B5_body_encoding_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```

并行运行时，把run改成`body_encoding_three_parallel_v1`。报告保留配置、日志、初始化检查、截断诊断、模板CSV和评分，不打包波形缓存或权重。
