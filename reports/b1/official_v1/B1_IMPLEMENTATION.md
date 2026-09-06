# B1 实现

## 一、完成情况

B1 使用 Lightweight 1D U-Net，独立实现于 `b1/`，读取 main 的公共数据契约、固定 subject 划分、统一预处理、loss 和 V0 评价接口。代码不修改 main 的公共文件，不修改原始数据，也不包含 adapter 或 baseline head。

正式矩阵按共享文档固定为 14 组：task1 的 A_L0、A_L1、B_L0、B_L1、C1、C2；task2 的数据 A/B × 路线 A/B × L0/L1。C1/C2 使用固定质量加权 R 峰伪配对，task2 不设置 C 路线。

代码分支：`baseline/B1`  
实现提交：`6718bfd`

## 二、模型与统一协议

模型为四级 1D U-Net，通道 16/32/64/128，卷积 kernel=3、padding=1，三次二倍下采样，线性插值后与 skip connection 拼接，输出 12 导联线性层。task1/task2 参数量分别为 163004/163244。首轮比较不使用 normalization、dropout、adapter 或 baseline head。

所有实验固定 seed=42、deterministic、batch size=16、AdamW、learning rate=0.001、weight decay=0.0001、无 scheduler、无 gradient clipping、无 early stopping；每个训练阶段 100 epochs。官方 checkpoint 只按 validation raw-uV V0 选择，task1 使用 r1，task2 使用 r2。

每个实验同时保存 E1（纯预测 d12）和 E2（按协议将输入中已有导联覆盖回预测后再评分）；task2 另保存 machine/body 的 subject-macro 结果。centered 指标只作形态诊断。

## 三、交付文件

代码、配置、只读检查和运行说明已经包含在 B1 分支。正式运行目录 `results/b1/official_v1/` 被 `.gitignore` 排除；完成全部实验后只将 `delivery/` 下的小型 CSV/Markdown 汇总复制到本目录。禁止提交原始 ECG、NPY/NPZ、预测文件、权重和大日志。

正式运行命令：

```bash
python -m b1.check_official_v1
python -m b1.run_official_matrix --device cuda
python -m b1.summarize_official
```

## 四、当前状态

正式矩阵已在 AutoDL 独立工作副本启动，结果位于 `/root/autodl-tmp/huawei_B1_official/results/b1/official_v1/`。AutoDL SSH 会话中断时，使用各实验目录的 `checkpoint_latest.pt` 续跑；在所有 14 组完成前，不将结果表标记为最终结果。
