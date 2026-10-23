# B5-1022：独立与成对 K16 推理复评

## 目的

对当前完整 B5-1022 进行**只推理、不训练**的采样策略对照：

```text
FM + P 基础生成
  → G120 全局慢修正
  → Task2 机器 FIR25 或体脂秤 C3 修正
  → 可靠 I / 目标导联 QC 的完整 Task1、Task2 平均 r
```

`independent` 是原始 16 份独立噪声；`antithetic` 使用 8 对 `+ε/-ε`，仍是
16 次 Heun16 rollout 后逐点平均。两者都保持 K=16，不能把 16 个相关系数再平均。

本入口不会更新 FM、P、G120 或 C3 权重。

## 必须使用同一套 B5-1022 权重

需要四项路径：

|参数|应填写的内容|
|---|---|
|`--slow-run`|含 `study.json` 的 B5-1022 `shared/` 目录|
|`--g120-checkpoint`|该 study 对应的 `G120/best.pt`|
|`--c3-cache`|生成当前最佳 C3 时使用的 `cache/` 目录|
|`--c3-checkpoint`|当前最佳 C3 的 `best.pt`|

`c3-cache` 与 `c3-checkpoint` 必须来自同一 B5-1022 C3 实验。脚本会校验缓存和权重协议；不应为了运行方便改用旧 B5-final 的 body 权重。

## 先做同噪声的独立/成对对照

在服务器 B5-1022 工作树、BioFlow 环境中定义当前最佳模型的实际路径：

```bash
cd /home/qht/huawei_B5_1022
conda activate BioFlow

slow_run=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared
g120=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/G120/best.pt
c3_cache=/填写当前B5-1022最佳C3对应的cache目录
c3=/填写当前B5-1022最佳C3的best.pt
```

先运行独立 K16（这是新噪声组的对照）：

```bash
bash scripts/run_b5_1022_noise_probe.sh \
  --gpu 1 --mode independent --noise-scale 1.0 --noise-seed 42 \
  --slow-run "$slow_run" --g120-checkpoint "$g120" \
  --c3-cache "$c3_cache" --c3-checkpoint "$c3"
```

完成后才运行相同 seed 的成对 K16：

```bash
bash scripts/run_b5_1022_noise_probe.sh \
  --gpu 1 --mode antithetic --noise-scale 1.0 --noise-seed 42 \
  --slow-run "$slow_run" --g120-checkpoint "$g120" \
  --c3-cache "$c3_cache" --c3-checkpoint "$c3"
```

每轮最终读：

```text
results/B5-1022/noise_probes/.../final/summary.csv
```

其中 `mean_r` 是最终比较指标。

## 换噪声组复核

当前 probe 支持 `--noise-seed`。复核时必须对同一新 seed 同时跑独立和成对：

```bash
# independent seed43
... --mode independent --noise-scale 1.0 --noise-seed 43 ...

# antithetic seed43
... --mode antithetic --noise-scale 1.0 --noise-seed 43 ...
```

比较每个 seed 内的 `antithetic − independent`，不能跨 seed 只挑最高分。若两个 seed 都为正，说明成对噪声收益不依赖 seed42 的偶然组合。
