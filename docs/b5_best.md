# B5-best：原B5快残差＋S50-F预测慢曲线

本分支从稳定B5 `f8ebe1b` 创建。main、B5-S50和正在运行的续训不受影响。当前入口不是瓶颈条件接入，而是两个冻结预测器的输出组合。只按 `(Task1 r + Task2 r)/2` 比较，RMSE不算分。

## 1. 固定的当前方案

```text
测试可见的120秒同步I＋人口学
  ├─ 原B5 EMA，Heun16/K16/seed42 → 12导联预测 p
  └─ 冻结S50-F，前后各20秒I → 12个中心10秒 → 拼成预测慢曲线 s
完整输出（缺失11导联）= p - 501点居中移动平均(p) + 0.75 × s
```

全部移动平均和50→500Hz插值在完整120秒坐标上做，端点复制。不逐窗去均值、不用真实目标均值或慢标签、不额外放大。I输出保持原FM的I头，与历史评估一致；两个任务只计其余11导联。S50看的是前后同步I，不是设备历史context；需要完整120秒输入，不是实时因果推理。

默认源文件在 `configs/b5_best.json`，并校验原FM与S50-F的SHA。原FM是第45轮 `best.pt`，不是最近续训的S50接入权重。S50-F是原慢CNN实验的 `best_score.pt`；这个旧文件名不代表本分支继续按bonus选模。

当前固定`slow_beta=0.75`，应用于全部记录的缺失11导联；不按患者真值选择系数。新导出采用v2模型包，推理和验证从包内读取该参数。默认采样仍为Heun16/K16/seed42。

当前QC参考：Task1 r=0.69816428，Task2 r=0.69167254，平均r=0.69491841。seed43复核为0.69475046；seed42的beta=0.5为0.68887223。0.75是目前已测试候选中更好的选择，不是新患者上的保证。

旧v1模型包仍按beta=1运行，其参考为Task1 r=0.69585536、Task2 r=0.68940334、平均r=0.69262935。更新代码不会改变旧模型包的输出；需要重新导出v2包才能默认使用0.75。网络、权重、条件、损失均未改变，不用重新训练。详细操作见`docs/b5_best_beta075.md`。本机没有真实源权重，不能把合成测试当作服务器ECG复现。

## 2. 保存与提交后，服务器建立独立worktree

本机推送（本次提交不会自动推送）：

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best'
git push --set-upstream origin B5-best
```

服务器远程按已有gitee配置使用，需先确保它包含B5-best：

```bash
cd /home/qht/huawei
git fetch gitee B5-best
git worktree add --detach /home/qht/huawei_B5_best FETCH_HEAD
cd /home/qht/huawei_B5_best
conda activate BioFlow
```

worktree已存在时，只在该目录 `git fetch gitee B5-best`、`git switch --detach FETCH_HEAD` 更新。不要切换正在训练的worktree。

## 3. 导出独立模型＋验证复现（不训练）

原FM、原S50-F及原prepared/prepared.json需仍存在。**导出只读取prepared.json的尺度和来源绑定，不读train/validation.npz或目标波形。** 导出的model.pt内嵌两个网络、尺度和来源信息，之后新记录推理不需要任何原checkpoint路径或prepared文件。

```bash
cd /home/qht/huawei_B5_best
sed -i 's/\r$//' scripts/run_b5_best.sh
bash -n scripts/run_b5_best.sh
bash scripts/run_b5_best.sh --gpu 1
```

默认结果：`/home/qht/huawei/results/B5-best/stable_beta075_v2/`。脚本不联网、不fetch、不训练。验证读取原huawei目录的配置和数据，因此不用在新worktree复制data、metadata或缓存。改变结果目录使用 `--root /home/qht/huawei/results/B5-best/stable_beta075_v2_repeat`。已有或中断目录不会覆盖；旧stable_v1目录保留。

成功条件：evaluation/evaluation.json中 completed=true、reference_matched=true；两项r与历史参考分别相差不超过2e-5。复现模式还核对人口学、QC、划分、配对表、窗口元数据和验证目标文件SHA。报告只使用correlation_score.json/csv/md，不输出60分制总分或bonus。逐导联RMSE可保留为诊断，但不参与排名。

复现失败时保留日志，核对源文件、QC、人口学和数值环境，不直接放宽容差或声称已经达到历史分数。

## 4. 新记录统一推理

下面输入路径是占位，需要换成真实I和人口学文件：

```bash
cd /home/qht/huawei_B5_best
CUDA_VISIBLE_DEVICES=1 python -m baselines.B5_best.predict \
  --checkpoint /home/qht/huawei/results/B5-best/stable_beta075_v2/model.pt \
  --anchor-npy /path/to/new_I_uV.npy \
  --metadata-json /path/to/new_metadata.json \
  --record-key new_record_001 \
  --output /home/qht/huawei/results/B5-best/new_record_001 \
  --device cuda
```

I必须为500Hz原始μV的[60000]或[1,60000]。人口学字段age、sex/gender、height、weight允许缺失，整个metadata-json也可省略。输出prediction_uV.npy为[12,60000]原始μV；默认Heun16/K16和slow_beta=0.75，全部设置已内嵌，不用额外传采样或组合参数。inference.json和evaluation.json都会记录slow_beta。稳定record-key用于固定噪声；复现验证时自动采用原target_record_id及窗口起点。

## 5. 更换候选组件与组合排名

当前版本支持兼容的原B5-U快残差模型＋S50/S50-F慢CNN。不同组件必须通过尺度和结构核验，使用显式 `--candidate` 导出到新路径，不覆盖stable_v1，也不沿用历史r声明。

```bash
python -m baselines.B5_best.bundle \
  --fm-checkpoint /path/to/candidate_B5_U/best_r.pt \
  --slow-checkpoint /path/to/S50_F/best_score.pt \
  --prepared-info /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared/prepared.json \
  --candidate \
  --output /home/qht/huawei/results/B5-best/candidate_001/model.pt
```

然后用与第3节相同的validate命令评估，候选不传 `--verify-reference`。**B5-S50的条件接入checkpoint结构不同，本入口会明确拒绝；后续支持它时必须添加对应加载器，不能把它直接当B5-U。** 本次不改正在续训的B5-S50。

排名使用组合完整输出，而不是FM单独输出的best_r；相同验证数据/质量mask、窗口顺序与Heun16/K16才能排名：

```bash
python -m baselines.B5_best.select \
  --evaluations /home/qht/huawei/results/B5-best/stable_beta075_v2/evaluation /home/qht/huawei/results/B5-best/candidate_001/evaluation \
  --output /home/qht/huawei/results/B5-best/ranking_001
```

selected.json记录获选模型的SHA和报告路径，不改写已有权重。只有新组合平均r更高，才将它作为下一版稳定组件。只能比较现存候选，不能声称遍历过所有历史epoch。

## 6. 本分支以后的B5-U训练保存规则

`baselines.B5.train_huawei`的网络、条件、FM损失、学习率和训练预算不变，只改选择指标为main_correlation。每次验证记录r1/r2/平均r，新运行只写best_r.pt、last.pt和训练历史；不创建best.pt或best_bonus.pt。不要根据新的r规则删除Huber等原重建损失。

新score_protocol=mean-r-target-record-qc-v1，旧bonus/total恢复状态不能resume；可用旧兼容B5-U权重作为 `--init-checkpoint` 在新结果目录开始。新last.pt才能精确resume。原B5-S50续训应继续使用它自己原来的代码，不受本分支影响。公开预训练不需要因为封装重跑。

base与两个main-v2实验配置已切换到新输出目录和平均r；共同ecg12gen/evaluate.py保持原样以保留历史可解释性，本分支封装其r聚合结果并使用独立的新评分入口，不把旧60分制字段当新分数。

## 7. 本地验证范围

58项CPU检查通过，1项需要真实华为缓存的检查因本机缺数据跳过。覆盖原B5模型/ODE/采样回归、绝对电压和QC聚合、两个组件打包与完整性校验、源文件移除后的新记录推理、S50的50秒窗口与中间裁剪、501点全记录组合、50→500Hz坐标、组合排名的数据一致性和best_r/last恢复状态。Bash语法与Git差异格式检查通过。

本次beta=0.75更新另通过25项B5-best CPU合成检查，覆盖新包参数持久化、源文件移除后的75μV/100μV新旧推理区别、无效系数拒绝、评估报告记录与旧beta=1诊断兼容。

没有启动训练或执行优化器更新。端到端文件流程检查使用合成数据和部分采样替身；真实FM的短信号ODE与原B5回归另测。0.694918参考来自已上传的服务器报告，本机没有用真实权重复现，也未验证服务器GPU显存或耗时；先运行第3节复现检查再用于正式测试。
