# 体脂秤context：历史编码配对实验

本次只验证历史特征编码，不加入门控，不改变FM、慢CNN、d6、评分规则或当前I/原预测II的编码方式。旧编码和旧权重仍可原样使用，实验不会自动替换默认模型。

## 1. 两组做什么

|项目|O：原编码对照|E：形态＋幅度编码|
|---|---|---|
|历史I/II心搏模板|两条均除以I的RMS，截断到±8|分别除以各自RMS，再用±8保护极端尖峰|
|历史六导联关系|共享I尺度，先截断后计算21项相关性和6项对数RMS|截断前计算21项相关性，另提供6项相对I的对数RMS比例|
|输入尺寸|模板2×300，统计27维|相同|
|网络、初始化|原BodyIIAdapter，共享随机种子42|相同参数、相同初始权重|
|训练|10轮，batch16，AdamW，lr=0.0002|相同|
|损失|10秒PCC＋0.01×归一化修正量平方|相同|
|选best|完整120秒、QC正常I/II、目标记录等权的II相关性|相同|

模板先按历史I检测相同的事件，再取相同的最多32个心搏、逐点中位数。E的六导联幅度统计在完整有效历史10秒片段上计算后取中位数，不是逐心搏幅度标签。历史和当前目标属于不同时间，保持不对齐。E的log幅度比限定在±log(10000)，零幅度用1e-4数值保护。形态和幅度均来自测试可见的历史信号，不使用当前真实II。

本次两组从相同随机权重重新训练小修正器，**不是从C-best继续训练**。这是为了先检验输入编码本身；C延长训练属于另一个实验。默认10轮保持原body实验预算，可在新运行开始前统一指定`--epochs 20`，不能只增加一组预算。旧body训练和本轮成绩不能直接相减解释为编码收益，应比较同一轮的E与O。

## 2. 使用哪些现有结果

服务器代码工作树：`/home/qht/huawei_B5_best_context`。如果你的目录名不同，只需修改下面的`cd`。

旧缓存只读复用：

```text
/home/qht/huawei/results/B5-best-context/body_v1/cache
```

新实验目录：

```text
/home/qht/huawei/results/B5-best-context/body_encoding_v1/
  run.json
  encoded_cache/          # 新历史特征；I、B5预测、真实II、QC逐数组保持一致
  preflight_O*/           # 旧编码CPU检查
  preflight_E*/           # 新编码CPU检查
  preflight_pair.json     # 同初始权重、同预算、同数据的配对检查
  O/best.pt              # 旧编码新训练权重
  E/best.pt              # 新编码新训练权重，自带编码标识
  study/                 # 14组评估；实际目录记录在study_path.txt
```

无需重新生成B5预测，无需读取C/G的body权重。准备入口只重新读取原始历史，核对历史指纹并精确复现旧特征，再生成E缓存。原缓存不会改写。新特征维数不变，所以网络结构和参数数目也不变。

## 3. 更新代码

本地提交后由你推送。服务器在已有工作树执行：

```bash
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
```

工作树若本来是detached HEAD，`git merge --ff-only FETCH_HEAD`也可以快进到本次提交。若存在服务器上的本地改动，先检查并保存，不使用强制重置。

## 4. GPU1一条脚本依次完成准备、检查、两组训练和评估

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
sed -i 's/\r$//' scripts/run_b5_body_encoding.sh
bash scripts/run_b5_body_encoding.sh --gpu 1
```

默认配置文件：

```text
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
```

脚本顺序是：新编码缓存准备 → O/E CPU前向、反向、修正上限检查 → 同初始化配对检查 → O训练 → E训练 → 评估。日志在输出子目录之外，避免`tee`造成“输出目录非空”。不执行git fetch，不连接网络，不安装依赖。

中断后使用相同设置续跑：

```bash
bash scripts/run_b5_body_encoding.sh --gpu 1 --resume
```

重做新种子时使用新目录，B5推理噪声仍固定42：

```bash
bash scripts/run_b5_body_encoding.sh --gpu 1 --seed 43 \
  --run /home/qht/huawei/results/B5-best-context/body_encoding_seed43
```

## 5. 可选：O/E分两张卡训练

只选择本节或上一节一种操作，避免重复启动同一个输出目录。

先准备缓存：

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_encoding_parallel_v1
source_cache=/home/qht/huawei/results/B5-best-context/body_v1/cache
config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
mkdir -p "$run"
python -u -m baselines.B5_best_context.body_encoding_prepare \
  --config "$config" --cache "$source_cache" --output "$run/encoded_cache"

python -m baselines.B5_best_context.body_train \
  --cache "$source_cache" --group context --seed 42 --epochs 10 \
  --output "$run/preflight_O" --check-only
python -m baselines.B5_best_context.body_train \
  --cache "$run/encoded_cache" --group context --seed 42 --epochs 10 \
  --output "$run/preflight_E" --check-only
python -m baselines.B5_best_context.body_encoding_study \
  --config "$config" --cache "$source_cache" --encoded-cache "$run/encoded_cache" \
  --control-run "$run/preflight_O" --encoded-run "$run/preflight_E" \
  --check-only --output "$run/preflight_pair.json"
```

出现`ready: true`后，开两个终端，分别执行：

```bash
# 终端一：GPU0
cd /home/qht/huawei_B5_best_context
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5_best_context.body_train \
  --cache /home/qht/huawei/results/B5-best-context/body_v1/cache \
  --group context --seed 42 --epochs 10 --device cuda \
  --output /home/qht/huawei/results/B5-best-context/body_encoding_parallel_v1/O
```

```bash
# 终端二：GPU2
cd /home/qht/huawei_B5_best_context
CUDA_VISIBLE_DEVICES=2 python -u -m baselines.B5_best_context.body_train \
  --cache /home/qht/huawei/results/B5-best-context/body_encoding_parallel_v1/encoded_cache \
  --group context --seed 42 --epochs 10 --device cuda \
  --output /home/qht/huawei/results/B5-best-context/body_encoding_parallel_v1/E
```

两组都出现`completed.json`后评估：

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_encoding_parallel_v1
CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_encoding_study \
  --config /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml \
  --cache /home/qht/huawei/results/B5-best-context/body_v1/cache \
  --encoded-cache "$run/encoded_cache" --control-run "$run/O" --encoded-run "$run/E" \
  --output "$run/study" --device cuda
```

手动训练中断可在原训练命令后加`--resume`，参数不变。评估输出必须是空目录；重算时使用新目录名。

## 6. 怎么判断有效

- `E_both − O_both`：新编码对最终平均r的配对收益，两组都开启相同d6。
- `E_body − O_body`：只看body路线编码差异。Task1和机器组必须保持一致。
- O/E各自的`disabled`和三个固定`shuffled`：检查新编码是否更好利用个人历史，不能挑最有利的打乱结果。
- `body_encoding_deltas.csv`：看改善/下降是否更均衡，特别检查原先HNU11119、HNU11104的下降是否减轻，同时保住HNU11135、HNU11141的收益。
- `body_component_metrics.csv`：II整体r、快r、慢r和修正幅度；RMSE作为诊断，不作为本次选模指标。
- `encoded_cache/clipping_comparison.csv`、`templates/*.csv`：检查两组历史特征截断程度。O/E归一化尺度不同，模板纵轴不能按同一μV直接比较，也不能仅凭截断变少宣称ECG预测更好。

14组评估均复用同一基础预测。编码身份、缓存身份、实际训练初始权重及预算均会核对；不满足时直接报错。完成标记为`study/encoding_checkpoints.json`。旧权重不含编码标识时自动使用旧编码；E权重在实时推理时自动提取新编码，不需要额外传参数。不要把E权重交给未更新的旧代码。

## 7. 打包报告

一条脚本方式完成后：

```bash
run=/home/qht/huawei/results/B5-best-context/body_encoding_v1
tar -czf /home/qht/B5_body_encoding_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```

如果采用并行方式，将`run`换成`body_encoding_parallel_v1`。报告保留日志、配置、初始化检查、截断诊断、模板CSV和完整评分，不打包波形缓存或权重。
