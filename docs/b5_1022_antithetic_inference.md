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

脚本内置当前 B5-1022 的四项已核实路径：

|参数|应填写的内容|
|---|---|
|资源|默认位置|
|---|---|
|`shared/`|`/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared`|
|G120|`/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/G120/best.pt`|
|G120-C3 特征缓存|`/home/qht/huawei/results/B5-1022/g120_c3_seed42_20261022_195813/cache`|
|C3 权重|本仓库的 `models/body_c3_seed42_best.pt`|

`c3-cache` 必须对应本次冻结的 G120 `best.pt`，其中保存了将 C3 应用于该 G120 输出所需的可见历史特征。`c3-checkpoint` 可以是协议兼容、预先确定的 C3 权重（例如当前 B5-1022 所用的 `body_c3_seed42_best.pt`）；它不需要与这个 G120 特征缓存来自同一目录。脚本会校验 G120 缓存和 C3 权重协议，不能改用旧的非 C3 `body.pt`。

## 先做同噪声的独立/成对对照

在服务器 B5-1022 工作树、BioFlow 环境中直接运行即可。仅在服务器路径改变时，才通过参数覆盖默认值。

```bash
cd /home/qht/huawei_B5_1022
conda activate BioFlow
```

先运行独立 K16（这是新噪声组的对照）：

```bash
bash scripts/run_b5_1022_noise_probe.sh \
  --gpu 1 --mode independent --noise-scale 1.0 --noise-seed 42
```

完成后才运行相同 seed 的成对 K16：

```bash
bash scripts/run_b5_1022_noise_probe.sh \
  --gpu 1 --mode antithetic --noise-scale 1.0 --noise-seed 42
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
