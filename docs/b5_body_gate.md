# 体脂秤C/G门控实验与模板截断核查

## 1. 本轮具体改什么

只改体脂秤II修正器。心电图机d6 FIR25、B5-best的FM与慢CNN、人口学、β=0.75、Heun16/K16/推理seed42均保持原设置。复用body_v1/cache，不重新生成任何B5预测。

```text
原B5-best II + alpha × body修正量
```

|项目|C：继续训练对照|G：门控实验|
|---|---|---|
|初始化|同一body_v1/context/best.pt|同一权重|
|alpha|固定1|由可见输入预测，范围0～1|
|新增训练|原body CNN继续训练|原body CNN与门控一起训练|
|预算|10轮，seed42，lr0.0002，batch16|相同|
|损失|原10秒II PCC＋0.01×实际归一化修正量平方|相同|
|选模|完整120秒、可靠I与正常II的目标记录等权r|相同|

这次不增加完整记录PCC损失、不改变模板编码、不扩展五导联。固定1的C和学习alpha的G都有相同的门控参数初始化；C冻结这部分参数。

门控读取：整条可见当前I、原B5预测II的统计量，以及现有历史心拍模板和27维关系特征。总计48维，训练集拟合归一化，验证/测试复用。没有真实II、真实均值、目标质量标记、患者ID或验证成绩输入。特征中的模板平顶指标只是一个可见形态统计，不代表已经确认原数据有质量问题。

每条120秒记录只产生一个alpha，12个10秒窗口共用。门控初始alpha严格为1，保留原body输出，CPU预检会核对。实现为 `min(1, 2*sigmoid(z))`，最终投影零初始化；预检检查在边界仍有衰减梯度。若训练后alpha全为1，说明门控未学会有效衰减，不能宣称解决问题。

固定源权重：

```text
/home/qht/huawei/results/B5-best-context/body_v1/context/best.pt
seed42，原epoch10
SHA256: 48700ef893c714de4ba16627e0bc0ade52ba7c30ac2dc73b50ff4028c8f2beec
```

## 2. 截断核查是独立报告

对原始训练/验证历史重新提取**同样的**body特征，核对结果与原cache逐值一致。统计被选中心拍在归一化后、±8截断前的峰值及I/II截断比例。对HNU11135/HNU11141/HNU11119/HNU11104输出截断前后的中位模板CSV。

这只核查“特征处理是否裁平了波形”，不修改原始信号、不删记录、不改QC、不改变本轮训练输入。即使有截断，也需要结合结果判断其影响；不能仅凭平顶认定原始设备采集饱和或它就是r下降原因。

## 3. 服务器更新

本地提交推送B5-best-context后：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git switch --detach gitee/B5-best-context
git log -1 --oneline
```

有你自己的未提交修改时，先保留再更新，不要强制覆盖。只更新此工作树，不切换原仓库当前分支。

## 4. 推荐GPU1串行完成

```bash
cd /home/qht/huawei_B5_best_context
sed -i 's/\r$//' scripts/run_b5_body_gate.sh
bash -n scripts/run_b5_body_gate.sh
bash scripts/run_b5_body_gate.sh --gpu 1
```

默认输出：

```text
/home/qht/huawei/results/B5-best-context/body_gate_v1/
  clipping_audit/       全部记录截断比例＋四个案例模板CSV
  preflight_C/         C初始输出等价性检查
  preflight_G/         G输出等价、门控梯度、同记录alpha一致性检查
  C/                   对照best.pt、last.pt及逐轮记录
  G/                   门控best.pt、last.pt及逐轮记录
  study/               完整评分、逐记录alpha、波形分量诊断
  *.log
  *_path.txt           若中断重试，记录实际报告路径
```

脚本先做只读截断核查与预检，再依次训练C/G并评估；不启动FM训练，不重新采样B5。截断核查使用CPU和服务器原始历史数据，沿用SciPy依赖。

中断后同样路径续跑：

```bash
bash scripts/run_b5_body_gate.sh --gpu 1 --resume
```

已完成的核查保留，训练从last.pt恢复。模型初始化、训练seed、epochs、batch、学习率、缓存不能在续跑时改变。

## 5. 如果只想先完成截断核查

这是第4节的替代入口，不必重复跑。

```bash
cd /home/qht/huawei_B5_best_context
python -u -m baselines.B5_best_context.body_clip_audit \
  --config /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml \
  --cache /home/qht/huawei/results/B5-best-context/body_v1/cache \
  --output /home/qht/huawei/results/B5-best-context/body_clip_audit_v1
```

## 6. 结果怎么判断

study包含：baseline、仅d6、源body单独/加d6、C单独/加d6、G单独/加d6，以及G关闭历史、三个固定错配组。

1. **G_both − C_both**：门控的配对增量。不能只比较G与旧模型，因为继续训练本身也会改变结果。
2. **C_both − source_both**：原body继续训练的影响。
3. `body_gate_deltas.csv`：G相对C、源body的逐记录II r，以及快/慢分量变化。重点看旧下降患者是否减轻损害、旧改善患者是否保留收益。
4. `G_body/postprocess_records.csv`：每个体脂秤配对的record_alpha，整条120秒共用，不按目标挑选。
5. `gate_effect.json`：alpha分布、等于1的数量、G相对C升降记录数。alpha更小不自动意味着预测更好。
6. `clipping_audit/clipping_records.csv`：截断是否集中在下降案例，原模板特征有没有明显信息损失。这是诊断，不能根据验证患者ID设置门控名单。

机器组、Task1、胸导联应不受body权重变化影响；评分器检查禁止改动的导联差值为0，并检查设备路由增量可加。

如果G没有重复收益或明显损害局部形态，门控假设未得到支持。若核查确认大量有效峰形被裁平，再单独做“形状与幅度比拆分”的特征对照。本轮不将新权重自动设为默认。

## 7. 推理接入

旧 `baselines.B5_best_context.predict --body-checkpoint ...` 已兼容C/G权重，自动读取门控及训练归一化。仍只需要当前I、人口学、合法历史；不需要目标。机器FIR25保持原路径。未传body权重时，默认行为保持原样。

## 8. 打包

```bash
run=/home/qht/huawei/results/B5-best-context/body_gate_v1
tar -czf /home/qht/B5_body_gate_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```

本地不启动训练。CPU合成/静态检查不能代替服务器真实前向、门控训练及性能评估，脚本已提供训练前预检。
