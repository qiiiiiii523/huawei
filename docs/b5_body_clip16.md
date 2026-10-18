# 体脂秤context：在当前B5-best验证心搏16、统计8

本次只增加一组O16，不重训已有O/E/E0，也不把队友模型的绝对r与当前B5-best比较。

|项目|已有O对照|新增O16|
|---|---|---|
|历史I/II模板归一化|均除以历史I的RMS|相同|
|历史心搏模板限幅|±8|±16|
|历史六导联关系统计|共享I尺度、±8截断后提取27维统计|逐值相同|
|当前I、原预测II编码|原有设置|相同|
|修正器、初始化、seed、预算、loss|原O训练设置|从相同随机权重训练，全部匹配O|

“16/8”在这里明确指**历史心搏模板16、历史统计波形8**。它不是Heun步数/K，也不是输出修正强度。O16不采用E的分别归一化：与原8/8对照只改一项，更接近队友截图的比较。如果队友实现还改变了其他内容，这一组只能验证上述明确定义的改动。

本轮FM、慢CNN、d6、QC、120秒拼接、β=0.75、Heun16/K16及基础预测均不变。只训练体脂秤II小修正器，保留原修正上限。新缓存只改变历史模板，入口会验证27维stats与原缓存完全相同。测试时权重自带`legacy_beat16_stats8_v1`标识，自动使用16/8特征，旧权重仍使用旧编码。

## 1. 默认复用的结果

```text
旧缓存：/home/qht/huawei/results/B5-best-context/body_v1/cache
已有O：/home/qht/huawei/results/B5-best-context/body_encoding_three_v1/O
新结果：/home/qht/huawei/results/B5-best-context/body_clip16_v1
```

脚本从已有O/config.json读取seed、轮数、batch和学习率，不需要手动重填。你这次报告对应seed42、10轮、batch16、lr=0.0002。它复用O的评估权重作为对照，**不会从O/best.pt继续训练O16**，而是核对O16随机初始化参数哈希与当时O相同。

## 2. 更新代码并用GPU1运行

先将本地提交推送，再在服务器执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD

sed -i 's/\r$//' scripts/run_b5_body_clip16.sh
bash scripts/run_b5_body_clip16.sh --gpu 1
```

如果服务器工作树路径不同，修改cd。存在本地改动时先检查保存，不强制重置。脚本顺序：核对已有O → 准备16/8历史缓存 → CPU前向/反向检查 → 同初始化与预算核对 → 只训练O16 → 原O与O16配对评估。不执行git或联网，不重新生成FM预测。

中断后续跑：

```bash
bash scripts/run_b5_body_clip16.sh --gpu 1 --resume
```

原O如果在其他目录，显式指定：

```bash
bash scripts/run_b5_body_clip16.sh --gpu 1 \
  --control-run /home/qht/huawei/results/B5-best-context/body_encoding_three_v1/O \
  --run /home/qht/huawei/results/B5-best-context/body_clip16_v1
```

不要拿C/G继续训练权重当O；脚本会拒绝初始化、预算或缓存不匹配的对照。

## 3. 看什么

- `study/summary.csv`：O16_both减O_both是16/8对当前d6+B5-best组合的增量。Task1和机器组应相同。
- `study/body_encoding_deltas.csv`：O16−O逐患者II r、快r、慢r，以及O16相对未修正预测的变化。
- O/O16各自的disabled与三个固定shuffled：检查个人历史收益是否保留，不挑最有利的打乱结果。
- `clip16_cache/clipping_comparison.csv`：`*_beat_exceeds_8_fraction`表示超过8的比例；`*_beat_clipped_fraction`按实际限幅统计，O16对应超过16的比例。`template_clip_limit`标记真实阈值。不要只凭截断减少判定预测更好。
- `templates/`：原先改善和下降患者的历史模板前后CSV。

评分入口共14组，包含baseline、d6_only、O/O16各自body/both/disabled/三次shuffled。完成标记为`study/encoding_checkpoints.json`。如果评估中断，脚本会保留原目录并在retry新目录完成，实际位置见`study_path.txt`。

首轮只验证seed42。若最终r有值得保留的提升，再匹配新的训练种子复验；不能直接与队友不同初始化的分数比较。当前默认权重不会自动切换。

## 4. 打包

```bash
run=/home/qht/huawei/results/B5-best-context/body_clip16_v1
tar -czf /home/qht/B5_body_clip16_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```
