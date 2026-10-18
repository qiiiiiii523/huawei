# 体脂秤C26固定修正强度诊断

## 本次只做什么

固定已完成的C第26轮best.pt，不训练门控、不更新body，不改变O/E/E0编码实验。五组复用同一次body预测，只改变外部修正乘数：

```text
最终II = 原B5-best II + alpha × (C预测II − 原B5-best II)
alpha = 0 / 0.25 / 0.5 / 0.75 / 1
```

每组所有患者共用一个alpha，每条120秒记录的12个窗口也共用它。计算使用原始μV，不去目标均值、不平移波形。其他导联不变，机器d6保持FIR25，Task1不变。FM和慢CNN不加载运行；Heun16/K16、seed42、β=0.75来自原始预测缓存。

本次必须使用旧编码缓存 `results/B5-best-context/body_v1/cache`。C26是旧编码权重，不能改用E/E0缓存。旧权重通过兼容接口继续使用旧编码；脚本显式拒绝G、E、E0权重和新编码缓存。

源权重是续训实验目录 `body_C_extend30_实际时间/C/best.pt`，不是原body_gate_v1/C/best.pt（第10轮），也不是last.pt（第30轮）。本次报告对应：

```text
selected epoch: 26
full120s QC II r: 0.7857594639028548
SHA256: f23902346c786e0bef06388b86ef45c5514880f332ca1e0ff0e84c42f5688d45
```

脚本会在所有`body_C_extend30_*`目录中按完成报告和实际文件SHA寻找这份权重。找不到或匹配多份时明确停止，需手动指定，不会默默挑最新目录。

## 服务器操作：GPU1

本地提交并由你推送后，服务器已有工作树中执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
```

存在自己的未提交修改时先检查保存，不使用强制重置。若编码实验正在使用这个工作树运行，等它结束再更新代码，或在另一个已更新工作树运行；本脚本不修改编码缓存、权重或结果。

启动：

```bash
cd /home/qht/huawei_B5_best_context
sed -i 's/\r$//' scripts/run_b5_body_alpha.sh
bash scripts/run_b5_body_alpha.sh --gpu 1
```

若需要手动指定权重，先用下面命令查看真实目录，再把实际路径填入`--checkpoint`：

```bash
ls -d /home/qht/huawei/results/B5-best-context/body_C_extend30_*/C/best.pt
```

```bash
bash scripts/run_b5_body_alpha.sh --gpu 1 \
  --checkpoint /home/qht/huawei/results/B5-best-context/body_C_extend30_实际时间/C/best.pt
```

默认同时生成四个案例的全记录和58～62秒细节图。可添加`--no-plots`只算分；CLI还可设置`--detail-start`和`--detail-seconds`。脚本不联网、不安装依赖、不执行git命令。中断后保留原结果，新运行使用新目录；本次没有训练状态需要resume。

## 检查和输出

最终会打印实际目录，例如：

```text
/home/qht/huawei/results/B5-best-context/body_alpha_C26_日期时间_进程号/
  run.log
  study/
    alpha_checkpoints.json   完成标记、源权重SHA、设置、实际前向次数
    summary.csv              五组完整评分及体脂秤II r
    summary.md
    record_alpha.csv         所有QC有效I/II记录的逐alpha结果
    oracle_summary.json      诊断上限，不可部署
    oracle_records.csv       目标记录级最佳alpha，仅事后诊断
    plots/                   四个固定案例，每个overview/detail两图
    baseline/ d6_only/ alpha_000/ alpha_025/ alpha_050/ alpha_075/ alpha_100/
```

在全部打分后进行严格核对；只有通过核对和图表生成才写最终完成标记：

- alpha=0必须复现d6_only完整评分。
- alpha=1的24条有效记录及II r必须复现C26的`validation/epoch_026.json`，容差1e-7。这里不是复现旧C10的平均r0.702421。
- 五组的Task1和机器分必须相同；路由检查确保体脂秤仅II改变。
- 源缓存/权重、原预测与目标/QC/可见历史均核对。body的同一可见输入只前向一次，其他alpha复用其修正。

oracle先在每个目标内平均重复配对，再选这个目标最佳alpha，最后目标等权汇总。未定义r的目标从oracle及其固定alpha比较集合同时排除并明确记录，避免集合不一致。oracle使用真实目标，只说明潜在空间，不能作为正式成绩、不能在测试时挑选患者alpha。

## 怎么判断

|结果|后续|
|---|---|
|一个较小固定alpha更好|保留固定系数候选，独立复验后再决定；不必马上增加网络|
|逐记录上限明显高于最佳固定alpha|存在自适应强度空间；再检查测试可见信息能否预测适合的强度|
|固定alpha与诊断上限都改善有限|暂停衰减门控，优先改善历史编码与修正方向|

对照全部24条有效记录，重点查看HNU11135/HNU11141的受益是否被削弱，以及HNU11119/HNU11104的下降是否减轻。病例用于解释，不制定验证患者名单开关。不自动更新任何默认模型或alpha。

## 打包

```bash
run=$(ls -dt /home/qht/huawei/results/B5-best-context/body_alpha_C26_* | head -n 1)
cat "$run/study/alpha_checkpoints.json"
tar -czf /home/qht/B5_body_alpha_C26_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```

本地只做CPU合成验证，没有实际训练或对服务器数据算分。
