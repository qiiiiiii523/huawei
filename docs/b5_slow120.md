# B5-final-exp-slow120：整段120秒慢修正实验

## 1. 这个分支做什么

从 `B5-final` 的 `9f3e962` 创建。对应优化文档的方向一：检查更远处的同步 I，能否帮助恢复慢曲线的起伏和长走势。

| 组 | 做法 | 是否训练 |
|---|---|---|
| G0 | 原 B5-final：公开FM-best45＋0.75×P慢CNN＋d6/body | 否 |
| G50 | 冻结G0的FM/P，在原慢曲线上加一个只看局部50秒的小修正 | 只训练新增支路 |
| G120 | 同一个小修正网络，改为看完整120秒 | 只训练新增支路 |

当前参考平均 r＝**0.7075136581**（Task1＝0.7081216802，Task2＝0.7069056360）。这是历史验证成绩，实验是否提分要等服务器运行。

最终输出：

```text
原 FM 快部分＋0.75×（原 P 慢曲线＋新增慢修正）
                    ↓
            原 d6 / body 分设备处理
```

程序复用固定基础模型的float32组合输出，再用float64计算并加入 `0.75×新增修正`，最后恢复float32。零修正时基础输出逐点相同，避免重新拆分/组合造成第0步数值变化。

G50与G120的参数量、初始化、训练患者、批次顺序、损失、学习率、更新预算和验证频率相同；区别是可见时间范围。两组都有新增网络，所以必须同时与G0比较。它们处理的token数和计算量不同，会分别记录耗时和显存。

## 2. 网络与默认预算

- 输入：原500 Hz同步 I 和原人口学信息；修正支路不读取目标导联、历史context或其他导联预测。
- 新增输入先用 `scipy.signal.resample_poly` 从500 Hz抗混叠降到5 Hz。G120有600个时间点；G50每个窗口250个时间点。
- G50先取50秒原始I窗口，再独立滤波；每10秒只输出中央10秒。首尾超出真实记录的部分重复端点，不引入其他记录，也不先全局滤波后裁剪。
- 网络：32通道、两层、4头时间注意力，人口学编码、固定正弦位置编码、11导联输出头。输出头全零初始化，无dropout。
- 5 Hz修正按真实采样位置插值到原50 Hz慢时间轴，再按B5原来的块中心插值恢复500 Hz。原P慢曲线保持原分辨率。
- 训练seed43；20轮；batch＝2条完整记录；AdamW，学习率1e-4，weight decay＝0.01，梯度裁剪1.0。第0、5、10、15、20轮验证，第0轮也参与best选择。
- 损失沿用P路线：慢Huber＋0.1×慢PCC＋2.5890171539×原F残差Huber（原生50 Hz宽51）＋0.1×完整组合PCC；不加新的多尺度损失。
- 训练损失计算基础组合；**选best时计算经过固定d6/body后的完整平均r**。body权重不续训，历史按原B策略只处理一次。

默认预算是第一轮实验起点，尚未证明最佳。需要改预算时，两组一同改变，并使用新的run目录。

## 3. 需要服务器保留哪些内容

原数据与结果根目录默认 `/home/qht/huawei`。沿用B5-final配置选定的两份权重：

```text
results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt
results/B5-best-context/newFM_body_adapt_v1/adapt/best.pt
```

代码检查完整SHA256，不按文件名猜测best。还需要：原训练数据、metadata、可靠I/QC文件、B5-final使用的run_config，以及原基础验证预测缓存。默认配置已指向这些服务器路径。

准备阶段会重新为**训练患者**生成一次固定FM/P组合预测，可能比新增支路训练更耗时；G50/G120共用这份缓存，不重复跑FM。验证优先复用已完成的新FM＋P缓存，程序核对实际观测I、目标、患者身份与QC。

缓存用真实连续120秒记录。训练记录先按物理记录去重并要求12个真实连续窗口，检查训练/验证患者不交叉；不把10秒记录拼成120秒，不使用验证患者更新权重。

运行环境沿用服务器的 `BioFlow`：Python、PyTorch≥2.4、NumPy、SciPy、PyYAML。代码本身不联网下载任何数据或模型。

## 4. 上传并运行

### 方法A：上传配套代码ZIP

本地交付的 `B5-final-exp-slow120_code.zip` 是本次提交的完整Git代码快照，不含权重、数据、未跟踪文件。上传到服务器 `/home/qht/`，解压到一个新目录：

```bash
unzip /home/qht/B5-final-exp-slow120_code.zip -d /home/qht/huawei_B5_final_exp_slow120
cd /home/qht/huawei_B5_final_exp_slow120
conda activate BioFlow
bash scripts/run_b5_slow120.sh --gpu 1
```

### 方法B：通过Git传分支

在本地仓库手动推送（本次只做了本地提交，没有推送）：

```powershell
git -C C:\Users\Ashley\Desktop\HW\huawei push origin B5-final-exp-slow120
```

服务器使用自己的远程名fetch该分支，再创建独立工作树。下面以远程名 `origin` 为例；服务器若叫 `gitee`，把两处origin换成gitee。

```bash
cd /home/qht/huawei
git fetch origin B5-final-exp-slow120
git worktree add --detach /home/qht/huawei_B5_final_exp_slow120 origin/B5-final-exp-slow120
cd /home/qht/huawei_B5_final_exp_slow120
conda activate BioFlow
bash scripts/run_b5_slow120.sh --gpu 1
```

每次创建新的结果目录，默认：

```text
/home/qht/huawei/results/B5-final-exp-slow120/run_时间_PID/
```

顺序：检查权重 → 复现G0四种context组合 → 准备共享缓存 → G50训练 → G120训练 → 实际best/last权重复评 → 输出比较表。

G0或两组第0轮未复现历史完整组合，程序会报错保留报告，不继续把本次结果当作正确基准。

### 常用参数

显式指定结果目录、训练seed和预算：

```bash
bash scripts/run_b5_slow120.sh --gpu 1 \
  --run /home/qht/huawei/results/B5-final-exp-slow120/slow120_seed43_v1 \
  --training-seed 43 --epochs 20 --validate-every 5 \
  --learning-rate 1e-4 --batch-records 2
```

默认基础缓存缺失时，用 `--regenerate-base` 从选定FM/P重新生成验证预测；若缓存已存在则无需加它。

```bash
bash scripts/run_b5_slow120.sh --gpu 1 --regenerate-base
```

只准备数据、不启动优化器：

```bash
bash scripts/run_b5_slow120.sh --gpu 1 --prepare-only
```

结果根目录搬迁用 `--repository-root`；数据配置搬迁用 `--config`；同内容权重搬迁用 `--checkpoint` 和 `--body-checkpoint`。权重SHA仍保持固定。

## 5. 中断后怎么继续

不要用主脚本重新指向已有run，它有意拒绝覆盖。若 `shared/study.json` 已完整生成，直接运行相应训练入口：

```bash
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run=/home/qht/huawei/results/B5-final-exp-slow120/填写本次真实run目录

# 如果G50已经有last.pt，接着训练；G50完成则跳过这一条。
python -u -m baselines.B5_slow120.train --run "$run/shared" --group G50 \
  --output "$run/G50" --resume "$run/G50/last.pt" --device cuda --execute-training

# G120尚未开始时，不传resume。
python -u -m baselines.B5_slow120.train --run "$run/shared" --group G120 \
  --output "$run/G120" --device cuda --execute-training
```

若G120也已有last.pt，同样加 `--resume "$run/G120/last.pt"`。续训保存优化器、数据打乱器和随机状态，从最后完整保存的轮次继续；保留第0轮best。相同CUDA设备数量和运行环境下续训，不用best.pt冒充last.pt。

准备缓存中断、没有完整study.json时，保留旧目录，另开新run。完成缓存后不要改代码、study.json或训练参数；程序会检查指纹。

训练完成后，手动重评两组best/last并汇总（下面的验证输出目录须未存在；已经完成的跳过）：

```bash
for group in G50 G120; do
  for candidate in best last; do
    python -u -m baselines.B5_slow120.evaluate --run "$run/shared" \
      --checkpoint "$run/$group/$candidate.pt" \
      --output "$run/verification/$group/$candidate" --device cuda
  done
done
python -m baselines.B5_slow120.summarize --run "$run" --output "$run/comparison"
```

## 6. 看哪些结果，怎样判断

| 文件 | 用途 |
|---|---|
| `comparison/summary.md`、`summary.csv` | G0、G50-best/last、G120-best/last的两个任务、两设备和完整平均r |
| `comparison/summary.json` | G120−G0、G120−G50，是否同时超过 |
| `G50/best.pt`、`G120/best.pt` | 按完整context后平均r选择的新增支路；best可能仍为第0轮 |
| 各组 `history.jsonl` | 训练损失、更新数、每轮训练耗时、CUDA峰值显存 |
| 验证目录 `pair_metrics.csv` | 完整120秒逐患者／配对／导联的正式r诊断 |
| 验证目录 `slow_diagnostics.csv` | 原P＋新修正的慢r、起伏分段相关、长走势相关、幅度比与RMSE |
| 验证目录 `context_records.csv` | 原context实际使用和回退情况 |

通过第一轮初筛须满足：**G120同时超过G0和G50**。仅比退步的G50好、仍低于G0，不算升级。汇总的1e-7仅用于排除数值误差，不代表统计显著。

再检查收益是否对应真实慢走势改善、是否只由少数患者带动；按患者聚合重复配对。有效后用新的run将训练seed改成42复核，推理噪声seed始终固定42。没有正收益时先暂停这套设计，不持续扫学习率和损失。

默认验证不长期保存大预测数组，保留正式评分、逐记录诊断和source_arrays.json。需要某个候选的数组时再单独生成：

```bash
python -m baselines.B5_slow120.evaluate --run "$run/shared" \
  --checkpoint "$run/G120/best.pt" --output "$run/export_G120_best" \
  --device cuda --keep-predictions
```

保存的预测位于各task的prediction_uV.npy；对应目标/QC路径记录在source_arrays.json。

## 7. 从真实权重做单条预测

提供的是可见原始I、人口学和可选原始历史，不需要训练/验证目标或缓存数组。`shared`中只读取study.json作结构和权重指纹检查。

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.B5_slow120.predict \
  --run "$run/shared" --correction "$run/G120/best.pt" \
  --anchor-npy /home/qht/input/current_I_uV.npy \
  --metadata-json /home/qht/input/demographics.json \
  --record-key 实际稳定记录ID --task-id task1 --input-type watch_ecg \
  --output "$run/single_record_new" --device cuda
```

Task2用 `--task-id task2`，设备为 `ecg_machine_d6` 或 `body_scale_d6`，并加 `--history-npy`、`--history-mask-npy`、`--history-record-id`。输入历史为原始未滤波六导联，格式与B5-final一致。先完成实验，不直接替换正式提交入口。

## 8. 打包结果带回本地

把run设成实际目录后执行：

```bash
tar -czf /home/qht/B5_slow120_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" assets.json preflight.log G0 G0.log shared/study.json shared/preflight.json \
  prepare.log G50 G120 verification comparison
```

这份报告不含患者波形或权重，适合带回分析。服务器仍需保留完整run，尤其是共享缓存、两组best.pt/last.pt和原B5-final权重。

## 9. 本地交付范围

按用户要求，后续不在本地启动测试或真实训练。服务器入口、断点续训、复评和诊断代码随分支保存；合成数据检查代码在 `baselines/B5_slow120/tests/`，如需运行，放在服务器执行：

```bash
python -m unittest baselines.B5_slow120.tests.test_slow120 -v
```

本地仓库原有未跟踪文件均保留。原未跟踪的 `.gitattributes` 与B5-final同名文件冲突，已原样保存在 `.gitattributes.before-slow120`；没有将这些个人文件加入实验提交。
