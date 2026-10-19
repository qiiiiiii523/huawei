# 固定新FM：从当前B43-best15继续适配body

## 1. 本轮固定内容与唯一训练对象

|组件|设置|
|---|---|
|完整基础模型|`/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt`|
|验证预测缓存|`/home/qht/huawei/results/B5-best-context/FM_P_best45_B43_v1/base_inference`|
|body初始化|`/home/qht/huawei/results/B5-best-context/P_body_AB_seed43_v1/B/best.pt`，原B43-best15|
|旧B特征|`/home/qht/huawei/results/B5-best-context/P_body_AB_v1/prepared/B_cache`|
|保持不变|d6 FIR25、B整记录0.2Hz处理、O16、body α＝1、Heun16/K16、推理seed42、慢β＝0.75、原损失和QC|
|本轮训练|只更新body，训练seed43，额外10轮，batch16，lr＝0.0001|

这是一个新的小微调阶段：从已有body模型参数初始化，优化器重新初始化，不带入旧优化器状态；学习率为原body训练0.0002的一半。附加轮次从0计数，0代表未更新的原B43参数，不是把旧15轮重跑或改成总计10轮。FM、P慢CNN和d6不训练。

新FM的body训练预测只生成一次，验证预测直接复用本次0.707499组合对应的基础缓存。旧B模板、统计、mask和可用性原样复用，不重新改变滤波或截断方式。缓存会检查新旧同步I、目标II、QC、患者/目标划分和原始历史指纹一致，只有预测II输入更换。

训练前CPU检查加载权重与原body输出完全相同，并检查反向传播与修正边界。每轮按完整120秒、正常I/II、目标记录等权的II r选best；由于只有body II变化、评分集合固定，这与完整组合平均r的改进方向一致。保留第0轮候选，若10轮都无提升，best仍为第0轮。

## 2. 更新服务器

本地提交推送后，在服务器执行：

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_best_context
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
git log -1 --oneline
sed -i 's/\r$//' scripts/run_b5_body_adapt.sh
bash -n scripts/run_b5_body_adapt.sh
)
```

## 3. 推荐：GPU1一条脚本完成准备、训练和评估

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_body_adapt.sh \
  --step all \
  --run /home/qht/huawei/results/B5-best-context/newFM_body_adapt_v1 \
  --gpu 1
)
```

准备新FM训练预测可能较慢，后续小body训练较快。脚本不联网、不改变Git分支、不覆盖原权重、不自动更新部署默认配置。无需再上传B数据，也不用重新做A/B训练。

## 4. 查看结果

```bash
run=/home/qht/huawei/results/B5-best-context/newFM_body_adapt_v1
study=$(cat "$run/study_path.txt")
cat "$study/summary.csv"
cat "$study/body_adapt_study.json"
```

|group|方案|
|---|---|
|baseline|新完整模型，不加context|
|d6_only|新完整模型＋d6|
|current_both|新完整模型＋d6＋原B43，当前约0.707499|
|adapted_both|新完整模型＋d6＋本轮body-best（含第0轮）|

主要比较`adapted_both−current_both`的`mean_r`。Task1以及机器组r必须保持不变。还会验证训练第0轮逐记录II r准确复现`current_both`，以及所选best的训练验证II r准确复现完整评估，避免训练缓存与部署处理不一致。

新权重路径是`newFM_body_adapt_v1/adapt/best.pt`，只在确认完整组合提高后再考虑替换。即使best轮次为0，它也会携带新基础模型的匹配元数据，但body参数保持原B43，不是一次有效训练提升。不要使用`last.pt`代替best。

## 5. 中断后继续

```bash
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_body_adapt.sh \
  --step all \
  --run /home/qht/huawei/results/B5-best-context/newFM_body_adapt_v1 \
  --gpu 1 \
  --resume
```

已完成记录缓存按指纹复用；训练恢复本轮优化器和随机状态，仍只到额外10轮。若需要分步，可将all改成prepare、train、evaluate依次执行，其他路径与参数完全保持一致。同一run不能同时启动两个训练进程。

## 6. 打包报告

```bash
tar -czf /home/qht/B5_newFM_body_adapt_reports.tar.gz \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-best-context \
  newFM_body_adapt_v1
```

生产推理仍传原始完整body历史，由权重里的B协议处理一次。当前隐藏目标不进入预测器；真实II仅作训练监督和评分，不作推理校准。旧A/B、门控和随机初始化入口保持原行为，新增适配必须显式提供`--adapt-body-checkpoint`。
