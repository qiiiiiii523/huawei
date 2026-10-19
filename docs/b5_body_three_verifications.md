# body context补充验证：三步操作

本轮只补三项，不改变FM、慢CNN、d6、QC、β=0.75、Heun16/K16或基础预测缓存。当前E30-best24仍是已知最高分候选，不自动切换部署权重。

|步骤|目的|运行内容|默认GPU|
|---|---|---|---|
|alpha|确认当前最高分E24用哪个修正强度|E24固定权重，alpha=0/0.25/0.5/0.75/1；不训练|0|
|o16|同预算比较E/O/O16|O16从第10轮last.pt续到总30轮；保留原best；与已有O30配对评估|1|
|seed43|复验新编码收益|O/E同seed43、同随机初始化，各训练30轮并评估|2|

三项独立，可以在三个终端并行。第3项与此前seed43的O/O16各10轮不是同一个实验；它复验的是O/E在30轮预算下的新编码收益。

## 0. 先推送并更新服务器代码

本地新增支持：E编码alpha扫描、独立目录续训准备、E-best24路径核对及三步脚本。旧C26脚本仍保持原行为。**不能用旧run_b5_body_alpha.sh直接扫描E，也不能把原body_train的epochs改成30后直接resume。**

本地提交由你推送后，服务器执行一次：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
sed -i 's/\r$//' scripts/run_b5_body_verification.sh
bash -n scripts/run_b5_body_verification.sh
```

若有服务器本地改动先检查保存，不强制重置。更新完成后再启动三项，不在训练期间切换代码。每个新终端都需激活BioFlow并进入工作树。

脚本会查找唯一符合下列报告的E目录：seed42、30轮完成、best_epoch=24、II r约0.786604，并核对实际best.pt哈希。默认O30在该E目录旁边的O子目录。缓存固定复用：

```text
/home/qht/huawei/results/B5-best-context/body_v1/cache
/home/qht/huawei/results/B5-best-context/body_encoding_three_v1/encoded_cache
/home/qht/huawei/results/B5-best-context/body_clip16_v1/clip16_cache
```

多份匹配或路径不符时会明确停止，不按“最新目录”猜。可在下面每条命令后加`--e-run /真实E目录 --o30-run /真实O目录`。目录指包含config.json、completed.json、best.pt和validation的那一层，脚本会打印最终选中的E权重。

## 1. E-best24固定alpha扫描：GPU0

在终端一执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
bash scripts/run_b5_body_verification.sh --step alpha --gpu 0
```

不训练、不生成FM波形，只复用E修正结果进行五种混合。使用E对应的新编码缓存，alpha=1必须复现E第24轮逐记录II r，alpha=0必须复现仅d6。四个固定病例同时输出图片。alpha扫描的oracle使用真实标签，仅为诊断，不能部署。

输出：

```text
/home/qht/huawei/results/B5-best-context/body_verification_v1/alpha/
    paths.json                       真实E/O30路径和源文件哈希
    study/summary.csv
    study/alpha_checkpoints.json      完成标记
    study/oracle_summary.json
    study/plots/
```

看summary.csv哪一行mean_r最高。如果仍为alpha=1，就保留完整修正；不要把此前C26上的系数结论直接当作E24的结果。

## 2. O16续训至总30轮并评估：GPU1

在终端二执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
bash scripts/run_b5_body_verification.sh --step o16 --gpu 1
```

原始源目录只读：

```text
/home/qht/huawei/results/B5-best-context/body_clip16_v1/O16/
```

脚本将原last.pt、best.pt和验证日志复制到新的O16目录，只把新目录配置预算从10改为30。last.pt内的Adam、DataLoader和Torch/CUDA随机状态由原训练入口恢复，接着执行第11～30轮；历史best也保留，因此后续轮次下降不会丢失原最佳。新目录不复制旧completed.json，必须真正完成30轮才允许评分。这里不从best.pt重置优化器，也不修改原10轮结果。

输出：

```text
/home/qht/huawei/results/B5-best-context/body_verification_v1/o16/
    O16/extension.json          源last/best/config/日志哈希、预算变更记录
    O16/best.pt
    O16/last.pt
    O16/completed.json
    study/summary.csv
    study/encoding_checkpoints.json
```

O16与已有O30的配置会进行配对检查：除历史编码及缓存哈希外，初始权重、seed、loss、预算等必须一致。评分后看O16_both−O_both，再将O16_both与已知E30约0.702460对比。

## 3. seed43的O/E各30轮并评估：GPU2

在终端三执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
bash scripts/run_b5_body_verification.sh --step seed43 --gpu 2
```

两组从相同seed43随机权重出发，先做两组CPU前向/反向检查及同初始化核对，再在GPU2上依次训练O30、E30，最后评估。默认batch16、lr=0.0002、损失及mask不变。FM推理噪声仍是缓存中的seed42，不重新生成。不会复用此前10轮seed43的best权重。

输出：

```text
/home/qht/huawei/results/B5-best-context/body_verification_v1/seed43/
    preflight_pair.json
    O/best.pt
    E/best.pt
    study/summary.csv
    study/body_encoding_deltas.csv
    study/encoding_checkpoints.json
```

看本seed内部E_both−O_both是否仍为正，再与seed42同预算增量比较。不能拿seed43的E直接减seed42的O来判断编码收益。

## 4. 中断后续跑

对中断的那一项，在原命令末尾加`--resume`，例如：

```bash
bash scripts/run_b5_body_verification.sh --step o16 --gpu 1 --resume
```

alpha没有训练状态，若评分中断，脚本保留原study并另建retry目录；实际结果位置始终写入该项的study_path.txt。训练中断时恢复新目录last.pt及随机状态，不重启已完成轮次。已完成项再次resume会保留完成结果。

想重新完整运行三项，使用新的`--run-root`；不要覆盖当前结果。所有步骤重复运行时必须保持源E/O30/缓存不变。

## 5. 查看和打包

逐项查看实际评分目录：

```bash
root=/home/qht/huawei/results/B5-best-context/body_verification_v1
for step in alpha o16 seed43; do
  if [ -f "$root/$step/study_path.txt" ]; then
    study=$(cat "$root/$step/study_path.txt")
    echo "$step: $study"
    cat "$study/summary.csv"
  fi
done
```

三项完成后打包：

```bash
tar -czf /home/qht/B5_body_three_verifications_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C /home/qht/huawei/results/B5-best-context/body_verification_v1 .
```

不需要打包FM、慢网络权重或基础波形缓存；报告已包含实际源路径、权重哈希、配置、训练曲线、alpha结果、逐记录指标及图表。
