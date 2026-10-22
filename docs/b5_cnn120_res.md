# B5-final-exp-cnn120-res：零初始化CNN120残差对照

本实验保留 `B5-final` 的 FM、窗口式 P-S50 与 context 输出，另加一条只读取完整120秒同步 I 的 CNN 修正支路。它不替换原 P，也不复用 G120 的注意力支路。

| 组 | 新增 dilation | 目的 |
|---|---|---|
| L120 | `(1, 1)` | 匹配参数量的局部 CNN 残差控制 |
| C120 | `(256, 256)` | 理论感受野约122.8秒的长程 CNN 残差 |

两组都从同一 P-S50 特征主干初始化并冻结该主干，只训练新增卷积块和独立 correction head。新增块的第二层与 correction head 均零初始化。

## 强制门禁

- 准备阶段要求 L120/C120 初始权重逐位一致，初始残差逐位为零。
- 每组 epoch 0 评估要求残差逐位为零、修正后的预测逐位等于 Final，并要求所有汇总分数与 Final 的差小于 `1e-7`。
- 任一检查失败立即停止，不允许用后续训练掩盖初始化偏差。

## 运行

```bash
cd /home/qht/huawei_B5_final_exp_cnn120_res
conda activate BioFlow
bash scripts/run_b5_cnn120_res.sh --gpu 1
```

脚本依次复现 G0、建立不可变缓存、训练 L120/C120、复评 best/last，并输出 `comparison/summary.md`。默认结果目录为：

```text
/home/qht/huawei/results/B5-final-exp-cnn120-res/run_时间_PID/
```

## 判定

只有 `C120-best > Final` 且 `C120-best > L120-best` 才进入多 seed 复核。若 C120 不超过 Final，则停止 CNN 长程残差路线，不直接进入全量微调或从头预训练。
