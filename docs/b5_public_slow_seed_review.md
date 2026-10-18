# M／P训练seed复核与早期加密验证

## 1. 这轮做什么

M和P都保留，本轮两组都重新训练seed43；不从各自获胜best继续训练，而是从第一次实验的同一个原始慢PCC20开始。

| 项目 | 本轮设置 |
|---|---|
| 训练随机seed | 43；改变记录抽样及训练随机过程 |
| FM推理噪声seed | 仍为42，固定缓存不重生成 |
| 数据划分、QC、数据缓存 | 完全复用第一次CMP_v1 |
| 原始初始化 | CMP_v1/shared/components.pt，不用M/P训练后的model.pt |
| M、P训练路线 | 与第一次相同，仍各1560＋1560次更新 |
| 学习率及损失 | 与第一次相同，不同时试更低学习率 |
| 正式选模 | 仍每390次更新，输出best.pt，供训练seed42/43公平比较 |
| 额外验证 | 仅阶段二前390次中每78次加测；新增78、156、234、312四个点 |
| 额外候选 | dense_best.pt：起点及阶段二全部已观察点中最好；与正式best分开 |
| 阶段一候选 | stage1_best.pt：保存阶段一验证最好的权重，供诊断，不改变正式选模 |

加密验证会保留／恢复随机状态，不应改变参数训练轨迹。本地有合成测试验证。它增加少量验证计算，不增加训练更新次数。

**先用正式best比较seed稳定性；dense_best用于判断原390步间隔是否漏掉早期高分。不能把seed43的加密best与seed42的稀疏best混合后宣称稳定胜出。**

第一轮M阶段一1170的旧权重若没有另行备份，不能补回；本轮会开始保留阶段一候选。

## 2. 更新代码

本地提交后自行push到origin，再按现有方式同步Gitee。服务器：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow

git status --short
git fetch gitee B5-best-exp-PublicSlow
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

本次更改了训练模块，所以使用下面全新的结果目录与study。不要覆盖CMP_v1，不要用新代码resume旧study；旧导出model.pt仍可用于独立推理。

## 3. 一次准备新study，不重处理PTB／不重算FM

```bash
cd /home/qht/huawei_B5_public_slow
set -o pipefail

run=/home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1
mkdir -p "$run"

python -u -m baselines.B5_public_slow.prepare \
  --bundle /home/qht/huawei/results/B5-slow-public/CMP_v1/shared/components.pt \
  --huawei-cache /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache \
  --cache-study /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/shared/study.json \
  --public /home/qht/huawei/results/B5-slow-public/CMP_v1/public \
  --output "$run/shared" \
  --train-seed 43 \
  --validate-every 390 \
  --dense-validate-every 78 \
  --dense-stage2-until 390 \
  2>&1 | tee "$run/preflight.log"
```

看到ready=true、training_seed=43、fixed_inference_sampling.seed=42后再训练。CPU校验缓存期间GPU没有活动是正常的。不要重复prepare；它不会覆盖已有study。

## 4. 两个终端并行训练

终端一：M，GPU1。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow

bash scripts/run_b5_public_slow.sh \
  --group M --gpu 1 \
  --run /home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1
```

终端二：P，GPU2。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow

bash scripts/run_b5_public_slow.sh \
  --group P --gpu 2 \
  --run /home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1
```

每组自动完成两个阶段并导出正式best的model.pt。中断后在对应原命令末尾加`--resume`。

## 5. 两组完成后汇总（不要求C组）

```bash
cd /home/qht/huawei_B5_public_slow
set -o pipefail
run=/home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1

python -m baselines.B5_public_slow.compare \
  --root "$run" --groups M P \
  --output "$run/comparison_coarse"

python -m baselines.B5_public_slow.compare \
  --root "$run" --groups M P --checkpoint dense_best.pt \
  --output "$run/comparison_dense"

python -m baselines.B5_public_slow.compare \
  --root "$run" --groups M P --checkpoint stage1_best.pt \
  --output "$run/comparison_stage1"

python -m baselines.B5_public_slow.review_seeds \
  --reference-root /home/qht/huawei/results/B5-slow-public/CMP_v1 \
  --root "$run" \
  --output "$run/seed_review"

cat "$run/seed_review/seed_comparison.md"
cat "$run/comparison_dense/summary.csv"
```

汇总文件已经存在时，不重复运行覆盖；选择新输出目录。

| 看什么 | 回答什么 |
|---|---|
| seed_review/seed_comparison.md | 原390步规则下，训练seed42/43的M/P平均r及排序是否一致 |
| comparison_dense/summary.csv 与 comparison_coarse/summary.csv | 同一seed43、同一训练过程中，加密早期验证有无找到更高候选 |
| comparison_stage1/summary.csv | 阶段一是否有值得保留的候选，尤其M |

seed复核会检查初始化包、数据缓存、采样、损失、学习率和预算相同，且不接收混入dense候选的结果。

两个训练seed仍不是独立测试或显著性检验。本轮未重跑C-seed43，主要回答M/P路线的重复性及排序；若要进一步验证公开数据相对华为对照的增量，可后续补C。

## 6. 导出额外候选（可选）

正式model.pt已经由训练脚本导出。需要复核加密候选时，另存文件，不能覆盖正式包：

```bash
cd /home/qht/huawei_B5_public_slow
run=/home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1

for group in M P; do
  python -m baselines.B5_public_slow.export \
    --study "$run/shared/study.json" \
    --checkpoint "$run/$group/dense_best.pt" \
    --output "$run/$group/model_dense.pt"
done
```

如加密候选有收益，再决定是否对seed42也按相同加密规则复跑，才能公平比较新的选模方案。第一轮不必先重跑seed42。

## 7. 打包给分析

```bash
run=/home/qht/huawei/results/B5-slow-public/CMP_seed43_review_v1

tar --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  -czf /home/qht/huawei/B5_public_slow_MP_seed43_review.tar.gz \
  -C "$run" shared M P M.log P.log preflight.log \
  comparison_coarse comparison_dense comparison_stage1 seed_review

ls -lh /home/qht/huawei/B5_public_slow_MP_seed43_review.tar.gz
```

最终服务器路径：`/home/qht/huawei/B5_public_slow_MP_seed43_review.tar.gz`。

本轮不替换当前部署方案；先看复核结果，再对拟部署的包进行完整生成validate。

本地已通过46项CPU合成及回归检查；没有启动真实ECG训练，尚无训练seed43的效果结论。
