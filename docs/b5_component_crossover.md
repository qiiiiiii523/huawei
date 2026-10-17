# B5组件交叉组合：服务器操作

## 做什么

分支仍为 **B5-best-exp-SlowPCC**，本地工作树：

```text
C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-SlowPCC
```

新增入口为`baselines.B5_cross`，不修改`B5_slow_pcc`、原FM、慢CNN、训练配置或旧结果。原study代码指纹不变，不重新prepare，不修复/重建旧缓存，也不训练。

评估完整六格，以四个已知成绩作为复现参照，重点补两个未知组合：

|FM快残差|原S50-F|SlowPCC对照best|SlowPCC实验best|
|---|---|---|---|
|原FM|复核0.694918|复核0.697168|复核0.697689|
|control100 FM|复核0.695537|**新增组合**|**新增组合**|

保持β=0.75、Heun16/K16、完整120秒、目标导联QC、目标记录/Task2两设备等权，按平均r比较。seed42复用已保存FM完整预测，每个慢CNN预测只算一次，两种FM共用。只有慢CNN推理与CPU评分，没有FM积分和优化器更新。

## 1. 本机推送

代码已由本次工作提交；你执行推送：

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-SlowPCC'
git push origin B5-best-exp-SlowPCC
```

若服务器通过gitee拉取，先确保该提交已经同步到gitee。

## 2. 服务器更新已有worktree

没有训练进程使用这个worktree时执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_exp_slow_pcc

git status --short
git fetch gitee B5-best-exp-SlowPCC
git merge --ff-only FETCH_HEAD
git log -1 --oneline

sed -i 's/\r$//' scripts/run_b5_cross.sh
bash -n scripts/run_b5_cross.sh
```

不需要新建worktree。若merge因本地文件冲突而停止，保留文件并处理冲突，不用reset --hard。

## 3. 先做seed42交叉组合，GPU1

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc

bash scripts/run_b5_cross.sh \
  --gpu 1 \
  --seed 42
```

脚本会自动创建新目录，例如：

```text
/home/qht/huawei/results/B5-best-cross/seed42_日期_时间_PID/
```

日志在结果目录旁边的`.log`文件里，不会再因为tee先生成日志而触发“输出目录非空”。校验文件期间GPU不动正常，随后是轻量慢CNN推理与CPU评分。

默认输入全部指向已知训练结果：

```text
SlowPCC实验根目录：
/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1
    shared/study.json
    shared/components.pt
    cache/cache.json
    cache/validation/*.npz
    control/best.pt
    pcc/best.pt

control100 FM权重：
/home/qht/huawei/results/B5-scratch/early_public_v1/control/checkpoints/last.pt

control100原始完整FM评估目录：
/home/qht/huawei/results/B5-scratch/early_public_v1/control/evaluations/last
    evaluation.json
    task1/prediction_uV.npy、window_metadata.csv、target_quality_mask.npy
    task2/prediction_uV.npy、window_metadata.csv、target_quality_mask.npy
```

评估目录中的target_uV.npy或target_reference.json指向的原目标文件必须保留。不要传`hybrid_slow_f`子目录，它已经是组合输出，不能再当完整FM来分解。

如果实际位置不同，显式指定，不要移动旧结果：

```bash
bash scripts/run_b5_cross.sh \
  --gpu 1 \
  --seed 42 \
  --slow-run /home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1 \
  --control-checkpoint /实际路径/control/checkpoints/last.pt \
  --control-evaluation /实际路径/control/evaluations/last
```

脚本核对control第100轮checkpoint与原评估SHA、模型结构、尺度、采样，按pair/target/时间对齐，并逐对检查目标与QC一致。重复目标保留各历史配对自身的FM预测，不强行要求它们逐点相同。四格已知成绩复现误差超过2e-5时停止，不直接宣布新候选成功。

每格保留预测/目标/QC数组，约需数GB磁盘空间；不会复制156条训练缓存。不要只删除原FM预测或validation缓存后再运行本实验。

## 4. 查看结果

运行结束时终端会打印完整结果目录。将下面run替换成它：

```bash
run=/home/qht/huawei/results/B5-best-cross/seed42_实际日期_时间_PID

cat "$run/summary.csv"
cat "$run/result.json"
cat "$run/report.md"
```

`result.json`必须为`completed: true`。`best_combination`与`portable_bundle`给出本次平均r最高的组合及可推理模型包。

命名：

- `original` FM：原B5。
- `control100` FM：scratch-control第100轮。
- `original` slow：原S50-F。
- `control20` slow：SlowPCC对照的best，实际轮次写在provenance。
- `pcc20` slow：SlowPCC实验的best，实际轮次写在provenance。

例如`control100__pcc20.pt`是真正的双模型推理包，不是训练checkpoint。包含正确FM EMA、慢CNN、尺度和β；测试只需要I及可用人口学，不依赖验证目标或训练缓存。候选包不会覆盖现有B5-best配置或默认模型。

## 5. 再做seed43噪声复核，GPU1

seed42成功后再运行：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc

bash scripts/run_b5_cross.sh \
  --gpu 1 \
  --seed 43
```

这一步要重新生成两种FM各40条目标的Heun16/K16预测，明显慢于seed42。每种FM、每条目标只生成一次，三种慢CNN共用同批FM输出。同步I明确读取原验证数据接口的可见anchor字段，并核对与冻结缓存的I/人口学一致；不将真实待补导联输入网络。

seed43是推理噪声复核，不是独立训练seed。它的模型包部署采样仍为seed42，不会把seed43参考成绩标成seed42复现标准。不要分别给候选挑最好seed，比较每个seed内部的同组差值。

## 6. 打包报告

短命令一次打包本目录所有交叉实验的小报告，不打包大数组与权重：

```bash
cd /home/qht/huawei

find results/B5-best-cross -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \) \
  -print0 | tar --null -T - -czf B5_cross_reports.tar.gz
```

发回`/home/qht/huawei/B5_cross_reports.tar.gz`即可分析。训练seed复现实验、V13模块作用诊断不在本次交叉组合入口里，也不会被自动启动。

## 本地验证

35项CPU合成功能及相关回归检查通过，Bash语法检查通过。覆盖EMA读取与导出、错组/尺度/指纹拦截、打乱行顺序的配对对齐、目标/QC变化拦截、配对自身FM预测保留、六格评分、慢CNN可见输入隔离、seed43输入与生成结果使用、模型包及代码指纹不变。

没有启动真实训练或GPU推理。合成测试数值不代表实验成绩，真实六格结果需按上述步骤在服务器生成。
