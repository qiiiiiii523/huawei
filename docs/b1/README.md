# B1 Lightweight 1D U-Net

正式实现读取 main 的公共 Dataset、预处理、loss、固定划分与 V0，不修改公共代码和原始数据。
当前分支只发布 official-v1 实验代码与交付汇总；旧原型保留在 Git 历史中，不放入当前交付目录。

## 正式实验矩阵

共 14 个训练实验：task1 为 A0/A1、B0/B1、C1、C2；task2 为数据版本 A/B × 路线 A/B × L0/L1。
C1/C2 使用固定、质量加权的 R 峰伪配对 loss，不重复构造 L0/L1。每个 checkpoint 同时评估 E1 和 E2。

## 运行与续跑

只读验收：`python -m b1.check_official_v1`。

隔离的最小冒烟检查：

`python -m b1.run_official_matrix --experiment B1_T1_C1 --smoke --max-batches 1 --device cuda`

正式运行或从 checkpoint 自动续跑：

`python -m b1.run_official_matrix --device cuda`

全部完成后生成交付表：`python -m b1.summarize_official`。

正式结果写到 Git 忽略的 `results/b1/official_v1/`。代码、配置和最终小型汇总文件提交到 B1 分支；checkpoint、预测 NPY 和原始数据不提交。

## 固定协议

seed 42、deterministic、batch 16、每阶段 100 epochs、AdamW、lr 0.001、weight decay 0.0001、无 scheduler、无梯度裁剪、无 early stopping。B/C 的 strict 预训练 checkpoint 按 task 复用；每个实验的适配阶段仍独立运行。B/C 适配保留 strict anchor：B 为 `L_sync + 0.20 × L_weak`，C 为 `L_sync + 0.20 × alignment_quality_score × L_pseudo`。

官方 checkpoint 只按 validation raw-uV V0 选择；task1 使用 r1，task2 使用 r2。E2 copy-at-eval、centered diagnostic 和 task2 分设备 subject-macro 均另列。

## 旧预检查

在仓库根目录执行 `python -m b1.preflight`。
结果写到 Git 忽略的 `results/b1/preflight/report.json`，没有 checkpoint。
该命令会执行合成输入的梯度更新，仅用于计时，不代表 ECG 模型训练结果。

## 固定初始结构

四级通道 16/32/64/128，每级两个 kernel=3、padding=1 的卷积和 ReLU。
三次 max-pool，下采样倍数2；线性插值到 skip 的确切长度再拼接卷积。
输出为12通道线性1×1卷积，不包含归一化层、dropout、device adapter、baseline head。
task1/task2 输入通道分别1/6；参数量163004/163244。
正式比较前冻结结构，不能在各组中静默变化。

## 数据与公共文件

`b1.data.local_config()` 读取原 common.yaml，仅在内存覆盖路径；支持当前同名嵌套目录。
没有修改公共配置、数据、subject split、原来的 huawei 项目。
预检查验证公共 Dataset 样本形状、有限值、门控、split，以及体脂秤 A/B canonical target 一致性。
strict 索引检查 train-only、dedup key 唯一和逐样本合同；尚不等同于全部数据治理审核。

## 2026-09-05 初步结果

task1 train/validation 882/213；task2 1311/300；strict train 962。
体脂秤 A/B 均为 train 350、validation 60。
CPU PyTorch 2.12.0+cpu，batch16，seed42，AdamW lr0.001、weight_decay0.0001。
8线程下，合成 Huber/PCC 训练批次中位耗时：task1 0.081秒，task2 0.083秒。
推理每批约0.023/0.022秒；采样进程驻留内存约0.6GiB（不是峰值）。
只测量4个预热后批次，不包含完整 loss、数据准备、V0 和文件写入。
不能据此承诺正式总耗时，也不能用作实验效果报告。
