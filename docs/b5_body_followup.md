# Body context：seed复验与波形诊断操作

这一步只做两件事：换一个body训练seed复验，以及画出现有/新权重的改善与下降案例。先不增加可靠性门控，不扩展五导联，也不重训B5-best。

## 1. 已有文件直接复用

```text
原实验：/home/qht/huawei/results/B5-best-context/body_v1
缓存：  /home/qht/huawei/results/B5-best-context/body_v1/cache
原权重：body_v1/context/best.pt、body_v1/no_context/best.pt
原评分：body_v1/study（若有study_path.txt，脚本优先按其中记录的位置读取）
```

缓存必须保留原 `.npz` 文件；之前打包给助手的报告不包含这些波形缓存，不能用报告压缩包替代服务器原目录。

**不需要重新运行body_prepare，也不需要重新生成训练集或验证集B5-best预测。**

新实验仍为同一网络、原损失、10轮、batch16、学习率0.0002和相同QC/120秒评分。只将body小修正器的初始化及训练打乱顺序从seed42改为seed43。两个配对组在seed43内部使用相同初始权重与顺序。

FM推理仍是seed42、Heun16/K16，β=0.75；不要把body训练seed和FM噪声seed混淆。旧seed42权重仍可读取。

## 2. 先推送本地分支，再更新服务器工作树

本地提交位于 `B5-best-context`。先用你原来的方式推送该分支；服务器执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context

git status --short
git fetch gitee B5-best-context
git switch --detach gitee/B5-best-context
git log -1 --oneline
```

如果status有你自己的修改，先保留修改再更新，不要强制覆盖。此操作只更新该工作树，不切换原 `/home/qht/huawei` 的当前分支。

## 3. 推荐：GPU1一条脚本完成

```bash
cd /home/qht/huawei_B5_best_context
sed -i 's/\r$//' scripts/run_b5_body_followup.sh
bash -n scripts/run_b5_body_followup.sh

bash scripts/run_b5_body_followup.sh --gpu 1
```

默认读取上面的 `body_v1`，写入全新目录：

```text
/home/qht/huawei/results/B5-best-context/body_seed43_v1
```

脚本按顺序完成：

1. 用seed42原权重、原缓存画图；重新计算24条有效记录II，核对与原study的II r一致。
2. CPU预检seed43的零初始化、前向/反向和修正幅度上限；没有优化器更新。
3. 在同一GPU依次训练seed43的no_context与context，均10轮。
4. 运行原来的四组与关闭/三次错配，共9组评估。
5. 对比seed42、seed43；检查基础B5预测、QC、两组训练预算相同，区别只有body训练seed。
6. 用seed43权重画相同四名指定患者的图，并额外选择一条接近零变化的记录。

CPU绘图需要Matplotlib；沿用BioFlow中的PyTorch、NumPy、PyYAML。脚本不联网、不安装依赖、不执行git命令。旧实验目录只读。

中断后同一路径续跑：

```bash
cd /home/qht/huawei_B5_best_context
bash scripts/run_b5_body_followup.sh --gpu 1 --resume
```

它会保留已完成的图和评估，训练从各自last.pt恢复。未完成的画图/评估目录会保留，再写新的retry目录。实际位置记录在 `*_path.txt`，不要删除旧实验或缓存。

如需显式指定路径：

```bash
bash scripts/run_b5_body_followup.sh \
  --gpu 1 --seed 43 \
  --source-run /home/qht/huawei/results/B5-best-context/body_v1 \
  --run /home/qht/huawei/results/B5-best-context/body_seed43_v1
```

## 4. 如果先只想看原实验的图

这一节是独立选项，不必在第3节之后重复执行。

```bash
cd /home/qht/huawei_B5_best_context
run=/home/qht/huawei/results/B5-best-context/body_v1
set -o pipefail

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_diagnose \
  --cache "$run/cache" \
  --study "$run/study" \
  --output "$run/diagnostics_v1" \
  --device cuda \
  2>&1 | tee "$run/diagnostics_v1.log"
```

如果实际study在别处，把 `--study` 改为 `cat "$run/study_path.txt"` 显示的路径。若diagnostics_v1已经存在，换一个新输出名；不要删数据重跑。

默认四名案例：

|患者|原实验用途|
|---|---|
|HNU11135|明显改善|
|HNU11141|明显改善|
|HNU11119|下降|
|HNU11104|下降|

再自动补一条变化接近零的记录，仅用于诊断。每条两张图：

- `*_overview.png`：120秒真实II、原预测、无context修正、正确context修正；501点慢分量；修正量与原预测误差。
- `*_detail.png`：相同58—62秒区间的可见I、原始II、快残差、修正量，以及历史I/II模板。

图中原始信号保持μV，不扣目标均值、不平移。快慢分量先在完整120秒计算，再裁切；这是诊断分解，不等于识别或删除生理基线漂移。历史模板来自另一次记录，使用历史I RMS归一化，不与当前目标逐点对齐。

若这段没有显示出关心的形态，可以另外输出固定区间，不重训：

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_best_context.body_diagnose \
  --cache "$run/cache" --study "$run/study" \
  --output "$run/diagnostics_38_42s" \
  --detail-start 38 --detail-seconds 4 --device cuda
```

所有图只用真实目标作事后诊断。目标不会进入修正器，也不会用于测试时挑选患者或调整强度。图只显示已缓存的历史模板，还不能证明不同历史片段的关系稳定；这属于观察结果之后决定是否补做的实验。

## 5. 运行完看哪里

```bash
run=/home/qht/huawei/results/B5-best-context/body_seed43_v1
cat "$(cat "$run/comparison_path.txt")/seed_comparison.csv"
cat "$(cat "$run/study_path.txt")/summary.csv"
cat "$run/original_plots_path.txt"
cat "$run/new_plots_path.txt"
```

主要看：

1. seed43下正确context相对无context、原输出的增益是否仍为正。
2. 相对已有d6的body额外增量是否重复出现。
3. 正确历史是否仍优于关闭和全部三组固定错配，不挑最好seed。
4. 两次实验的改善是否都集中在少数记录，下降案例具体损坏了哪部分波形。

不能将“选到了一个更高seed”当成稳定性证据。两个训练seed也不代替患者独立留出验证。先完成这些诊断，再决定门控、加长训练或五肢体联合修正。

## 6. 打包给助手

这次只打包新目录，包含两套诊断图、新训练记录、9组评分及seed比较；不包含原始缓存或权重：

```bash
run=/home/qht/huawei/results/B5-best-context/body_seed43_v1
tar -czf /home/qht/B5_body_followup_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" .
```

如果只做第4节画图：

```bash
run=/home/qht/huawei/results/B5-best-context/body_v1
tar -czf /home/qht/B5_body_plots.tar.gz \
  -C "$run" diagnostics_v1 diagnostics_v1.log
```
