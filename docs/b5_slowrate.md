# G120采样率实验：先选单Hz，再判断是否需要多Hz

## 为什么先做这个

原seed43的完整诊断表明，G120相对Final的平均r提高约0.014945，主要改善15秒以上趋势；短于5秒的诊断相关约0.243246→0.242641，中尺度约0.191793→0.195521。5Hz已经能表示几秒变化，不能断言短尺度问题来自采样率不足。

本轮是低成本检验“单一采样率选择是否限制当前分支”。它比直接增加多Hz网络更容易判断。P-S50适配仍可能有效，但它回答另一个问题，首轮不混入。

## 三组具体区别

|组|可见时长|输入率|token数|其余|
|---|---:|---:|---:|---|
|R2p5|120秒|2.5Hz|300|同原G120参数化|
|R5|120秒|5Hz|600|历史G120实现的匹配对照|
|R10|120秒|10Hz|1200|同原G120参数化|

- 三组原FM、P-S50、body/context固定，只训练新增注意力支路。标准模型各18,999参数。
- 每组从同一seed的新G120参数出发，输出头全零；不是从已训练G120-best继续。第0轮必须复现原Final。
- 同患者、批次顺序、损失、学习率、验证频率和优化器更新次数。训练预算默认继承原shared的study，可显式统一覆盖。
- 三组分别从原始500Hz观测I经`resample_poly`抗混叠降采样，禁止5Hz→10Hz插值冒充高率输入。
- 位置编码按真实时间换算为原5Hz时钟；5Hz分支的输入及非零输出与历史实现逐值一致。
- 修正输出按原50Hz块中心插值，最后使用原组合与context评分；标签、QC、scale和正式评分均不变。

采样率会同时改变抗混叠带宽、token密度、输出插值分辨率及计算量。本轮比较的是完整分支采样方案，不能据此把差异归因于某一个因素。固定的是更新预算，不是FLOPs。10Hz注意力矩阵约为5Hz的4倍，记录实际训练耗时和峰值显存。如果显存不足，三组都用同一个较小batch重新开run，不能只缩小一组。

## 输入与缓存

`--source-shared`必须是原`baselines.B5_slow120.prepare`生成的未改动shared目录，包含study.json、train和validation的npz。可用已完成的原seed43 run：

```text
/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared
```

准备阶段核对原代码、manifest和每份缓存SHA，仅生成3种可见I的轻量sidecar，不重新运行FM/P、不复制大波形数组。训练和复评仍依赖原shared及其中登记的config、baseline、权重，因此不要移动或修改这些源资产。新study独立记录训练seed和预算，不改旧study。

## 服务器启动：GPU1，tmux名slowrate

本地commit推送后，在服务器已有主仓库中拉取分支并创建独立工作树：

```bash
cd /home/qht/huawei
git fetch origin B5-final-exp-slowrate
git worktree add -b B5-final-exp-slowrate /home/qht/huawei_B5_final_exp_slowrate origin/B5-final-exp-slowrate
```

以上按服务器尚无同名本地分支/工作树写；已经存在时进入既有工作树核对commit，不要重复创建。

启动示例统一使用40轮、seed43，每5轮验证。40轮是本次统一预算，不代表已经证明最优；若采用其他预算，三组必须一致，最终与当前最好G120的预算和seed一起核对。

```bash
SLOWRATE_SOURCE=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared
SLOWRATE_RUN=/home/qht/huawei/results/B5-final-exp-slowrate/seed43_$(date +%Y%m%d_%H%M%S)

tmux new-session -d -s slowrate "cd /home/qht/huawei_B5_final_exp_slowrate && PATH=/home/qht/.conda/envs/BioFlow/bin:\$PATH bash scripts/run_b5_slowrate.sh --source-shared $SLOWRATE_SOURCE --gpu 1 --run $SLOWRATE_RUN --training-seed 43 --epochs 40 --validate-every 5; exec bash"

echo "结果目录：$SLOWRATE_RUN"
tmux attach -t slowrate
```

日志通过tee同时显示在tmux并写入run目录。脚本自动准备→R2p5训练→R5训练→R10训练→各自best/last实际权重复评→汇总。最后`exec bash`保留tmux供查看，不以会话存在与否判断训练是否还在运行。

退出tmux而保留运行：按`Ctrl-b`后按`d`。不需要轮询；稍后一次性查看：

```bash
tmux capture-pane -pt slowrate -S -40
```

完成后读取打印的`comparison/summary.md`。也可以只准备输入，添加`--prepare-only`；该选项不启动训练。

第二seed用相同设置、新run目录和`--training-seed 42`重做三组。先看单seed是否有清晰信号，再决定复核预算。

## 中断后继续

新开BioFlow终端，固定为首次打印的实际run目录。不要重跑启动脚本覆盖旧run；不要修改shared/study.json。

```bash
cd /home/qht/huawei_B5_final_exp_slowrate
conda activate BioFlow
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run=/上次打印的实际run目录
```

对已中断且存在last.pt的组运行（把R10换成实际组）：

```bash
python -u -m baselines.B5_slowrate.train --run "$run/shared" --group R10 --output "$run/R10" --resume "$run/R10/last.pt" --device cuda --execute-training
```

从未开始的组省略`--resume`；已完成的组不再训练。如果只创建了空输出目录而未写出last.pt，保留该目录作诊断，并为该组使用新的输出目录；汇总前需整理到本次run约定位置。不要拿best.pt冒充last.pt，续训会恢复优化器、数据顺序与随机状态。

三组完成后，对尚未成功复评的候选执行：

```bash
python -u -m baselines.B5_slowrate.evaluate --run "$run/shared" --checkpoint "$run/R10/best.pt" --output "$run/verification/R10/best" --device cuda
```

R2p5、R5、R10各做best和last。输出目录必须是新的；若旧目录是不完整复评，先保留/改名再重新使用标准位置。最后：

```bash
python -u -m baselines.B5_slowrate.summarize --run "$run" --output "$run/comparison"
```

## 怎么决定下一步

|观察|支持什么|下一步|
|2.5Hz稳定超过匹配5Hz|更强平滑/稀疏表示可能更合适|复核seed及最好G120的同预算比较，再考虑替换|
|10Hz稳定超过匹配5Hz|更高带宽/更密token可能有用|检查短中尺度是否真改善，再考虑提高采样率|
|5Hz仍最佳，其他组没有独有收益|原采样率已较合适|停止Hz扩展，转向剩余误差诊断或P适配|
|不同Hz在不同患者/导联有互补|存在融合候选线索|先用固定组合验证是否超过最佳单Hz，再决定多Hz网络|

单Hz分数不同不证明多Hz有效；在患者层面互补也只是线索，不能按验证患者真值选择模型。若试固定组合，需与同数量、同Hz的多seed集成比较，区分“多Hz”与一般集成收益，并在未参与该选择的数据上确认。

输出包括`summary.md/csv/json`、`patient_rate_contrasts.csv`，以及各verification的`pair_metrics.csv`与`slow_diagnostics.csv`。患者诊断按subject/task/device聚合，不把窗口当独立患者，也不把其简单均值冒充官方加权总分。训练history记录loss、更新次数、时间和显存；best保留第0轮资格，best/last均用实际权重复评。1e-7只排除数值误差，不是统计显著阈值。

本地合成测试不需要患者数据：

```bash
python -m unittest baselines.B5_slowrate.tests.test_slowrate -v
```
