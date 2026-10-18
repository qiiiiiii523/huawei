# B5-best-context：体脂秤 II 修正实验

## 1. 这次训练什么

冻结当前 B5-best（control100 FM＋SlowPCC20，β=0.75），用它的预测训练队友设计的小型 II 修正器。FM、慢CNN、人口学编码、Heun16/K16、seed42均不更新。不是重新训练B5-best，也不是把历史体脂秤波形注入FM。

```text
当前I＋人口学 → 固定B5-best → 120秒完整预测
                               ↓
机器记录：原d6一致性投影＋FIR25
体脂秤记录：当前I＋预测II＋历史六导联特征 → II小修正器
Task1：保持B5-best输出
```

默认基础模型由 `configs/b5_best.json` 指定：

```text
/home/qht/huawei/results/B5-best-cross/seed42_20261017_094115_417122/control100__pcc20.pt
SHA256: a8d32cd598f73a0d3335ff4453da9168eccbc518a9b508ae81fdb8babcce254b
```

缓存生成时会核对实际模型SHA与已完成验证的SHA。历史context属于同患者的独立时刻，不与当前目标逐点对齐。真实II仅用于训练标签和验证评分；推理接口不接收真实II、真实均值或目标质量mask。

## 2. 训练和评分

训练两组，同结构、seed42、相同初始权重、相同样本顺序和10轮预算：

|组|输入|作用|
|---|---|---|
|no_context|当前I＋B5-best预测II；关闭历史特征|通用II修正对照|
|context|上述输入＋历史六导联关系、历史I/II心拍模板|检查个人历史是否有额外收益|

复用队友的27维六导联关系特征、2×300心拍模板、小CNN/FiLM与±0.35×当前I RMS修正幅度上限。上限不是保证r提高。仅把预测残差加回原II，不去掉最终预测电压位置。当前I退化的窗口不修正。

训练损失：10秒II PCC＋0.01×归一化修正量平方；训练仅使用I、II质量正常的记录。保持原窗口训练目标，改用完整120秒、QC正常II的目标记录等权r选best；同一目标多配对先平均，不按窗口选best。保留epoch0原输出作为候选。每组有best.pt、last.pt、config.json、validation/epoch_*.json、completed.json。

这次只改变体脂秤II，其他10条待生成导联都固定；II在当前评分中的权重固定。因此在相同验证集合上，提高完整II r与提高最终平均r的选模排序一致。最终仍使用共享评分器核验全部导联、两设备等权与可靠I/目标导联QC，不用单导联数字代替最终成绩。RMSE保留诊断，不作为本轮选模目标。

缺失/禁用历史时仍保留已训练的通用II修正，和队友方法相同；它不是identity。完全禁用body修正应不传body权重。

## 3. 服务器更新工作树

本地提交并推送 `B5-best-context` 后，在服务器执行。不会切换 `/home/qht/huawei` 的当前分支。

```bash
cd /home/qht/huawei
git fetch gitee B5-best-context
```

若已经有此工作树：

```bash
cd /home/qht/huawei_B5_best_context
git status --short
git switch --detach gitee/B5-best-context
git log -1 --oneline
```

如果status有你自己的修改，先保留这些修改，再更新；不要强制覆盖。若尚未创建工作树，仅执行一次：

```bash
cd /home/qht/huawei
git worktree add --detach /home/qht/huawei_B5_best_context gitee/B5-best-context
```

运行环境使用原BioFlow；新增特征提取需要SciPy，基础依赖为PyTorch、NumPy、PyYAML。代码不下载模型，也不自动安装依赖。

## 4. 推荐：GPU1串行完成缓存、两组训练、全部评估

复用上次d6实验保存的原始B5-best `base_inference`，不是 `study/fir25`。脚本不联网、不git fetch。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context

sed -i 's/\r$//' scripts/run_b5_body_context.sh
bash -n scripts/run_b5_body_context.sh

bash scripts/run_b5_body_context.sh \
  --gpu 1 \
  --run /home/qht/huawei/results/B5-best-context/body_v1 \
  --baseline-evaluation /home/qht/huawei/results/B5-best-context/fir25_20261017_225742/base_inference
```

开始前可核对：

```bash
test -f /home/qht/huawei/results/B5-best-context/fir25_20261017_225742/base_inference/evaluation.json
python -c 'import torch, numpy, yaml, scipy; print("dependencies OK")'
```

缓存准备阶段会对训练目标从噪声生成完整120秒，再做快慢组合，计算量明显大于后续小网络训练。验证预测直接复用。输出逐记录进度，GPU不一定持续满载；历史特征提取及文件校验在CPU进行。

输出：

```text
/home/qht/huawei/results/B5-best-context/body_v1/
  cache/                  冻结训练/验证的I、预测II、标签II和历史特征
  preflight/preflight.json CPU零初始化、幅度界限、反向传播检查
  no_context/             无历史训练的对照权重
  context/                有历史训练的实验权重
  study/summary.csv        完整配对结果
  study/summary.md
  study_path.txt          记录本次评估的实际目录
  *.log                   日志放在上述子目录之外
```

中断后，**同一路径**加 `--resume`，不要重新创建另一套缓存或覆盖现有结果：

```bash
cd /home/qht/huawei_B5_best_context
bash scripts/run_b5_body_context.sh \
  --gpu 1 \
  --run /home/qht/huawei/results/B5-best-context/body_v1 \
  --baseline-evaluation /home/qht/huawei/results/B5-best-context/fir25_20261017_225742/base_inference \
  --resume
```

缓存按记录恢复；训练从last.pt恢复模型、优化器和随机状态。续训必须保持epochs、batch-size、学习率、缓存一致。若评估中断，将保留原目录并在新study_resume_*目录重做评估；实际位置始终见study_path.txt。

## 5. 想用两张卡并行训练：先完成缓存，再开两个终端

这一节是第4节的替代方案，不要与一键脚本同时跑同一路径。

终端一先准备缓存并完成预检：

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_parallel_v1
config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
mkdir -p "$run"
set -o pipefail

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_prepare \
  --config "$config" \
  --baseline-evaluation /home/qht/huawei/results/B5-best-context/fir25_20261017_225742/base_inference \
  --output "$run/cache" --device cuda \
  2>&1 | tee "$run/prepare.log"

python -m baselines.B5_best_context.body_train \
  --cache "$run/cache" --group context --output "$run/preflight" --check-only
```

看到 `ready: True` 后，两个终端分别运行；每个终端都要激活BioFlow并cd到工作树。

**GPU0，无context对照：**

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_parallel_v1
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5_best_context.body_train \
  --cache "$run/cache" --group no_context --output "$run/no_context" \
  --epochs 10 --device cuda \
  2>&1 | tee "$run/no_context.log"
```

**GPU2，有context实验：**

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_parallel_v1
set -o pipefail
CUDA_VISIBLE_DEVICES=2 python -u -m baselines.B5_best_context.body_train \
  --cache "$run/cache" --group context --output "$run/context" \
  --epochs 10 --device cuda \
  2>&1 | tee "$run/context.log"
```

两组都出现completed.json后再统一评估：

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_parallel_v1
config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_study \
  --config "$config" --cache "$run/cache" \
  --context-checkpoint "$run/context/best.pt" \
  --no-context-checkpoint "$run/no_context/best.pt" \
  --output "$run/study" --device cuda \
  2>&1 | tee "$run/study.log"
cat "$run/study/summary.csv"
```

## 6. 结果怎么看

|组|机器处理|体脂秤处理|
|---|---|---|
|baseline|无|无|
|d6_only|FIR25|无|
|body_only|无|正确context修正II|
|both|FIR25|正确context修正II|
|body_no_context|无|独立无context训练的II修正器|
|body_disabled|无|同一context模型，推理时关闭历史|
|body_shuffled_101/202/303|无|同一context模型，跨患者错配历史|

全部复用同一B5-best基础预测、同一QC与评分。Task1应完全不变；机器只改II到aVF；body只改II。每组的postprocess_records.csv保存修正幅度、历史是否可用、是否使用、回退与错配来源；pair_metrics.csv按导联检查。

1. `body_only − baseline`：整套body修正的收益。
2. `body_only − body_no_context`：相对通用修正器的额外收益。
3. 正确/关闭/三个打乱组：是否依赖正确个人信息。不要挑打乱种子。
4. `both − d6_only`：相对现有最佳d6方案的body增量。它应等于 `body_only − baseline`；脚本检查独立设备效果的可加性。

只改II，因此全局平均r的改善会被11导联、两设备、两任务平均稀释。不要把单II的r改善直接当作最终平均r改善。未验证前body不写入默认配置，原机器实验入口仍保留。

## 7. 测试时如何使用训练权重

原 `baselines.B5_best_context.predict` 增加可选参数：

```text
--body-checkpoint /home/qht/huawei/results/B5-best-context/body_v1/context/best.pt
```

与现有 `--input-type body_scale_d6`、`--history-npy`、`--history-mask-npy` 一起使用。加载会核对body权重所对应的基础B5-best SHA和采样设置。默认不传此参数，body保持原输出；机器FIR25不受影响。

## 8. 打包

一键脚本完成后执行以下短命令。不打包原始ECG、缓存和权重；保留训练记录、选模信息和全部评估：

```bash
run=/home/qht/huawei/results/B5-best-context/body_v1
tar -czf /home/qht/B5_body_context_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```

本实现仅在本地做静态/合成检查，不启动正式训练；真实性能以服务器以上配对实验为准。
