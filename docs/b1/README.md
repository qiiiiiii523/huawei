# B1 本地实现与计时

当前完成：初版网络、非破坏性本地路径接入、数据与合成 CPU 批次检查。
尚未完成：正式训练循环、八组实验调度、完整监督 loss 接线、checkpoint 与预测导出。

## 运行

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

## 待统一口径

八个主组合：task1 A/B/C1/C2；task2 数据A×路线A/B，数据B×路线A/B。
L0/L1 在普通弱配对分支有意义；C1/C2 当前公共配置的适配分支为伪配对，不能虚构两组差异。
copy-at-eval 是另列的评估对照，不覆盖原始 V0。
正式启动前需明确弱配对 observed consistency 合同、无 baseline head 的 raw 输出、
一致性/生理约束使用的共同尺度，以及 C2 的执行条件。
