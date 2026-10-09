# B5-U 首版代码验证记录

日期：2026-10-08。验证对象为同步 I＋人口学条件的一维 U-Net Flow Matching 首版。

## 已执行

- 19 项 CPU 单元/只读集成检查全部通过，耗时约 14.3 秒。
- 默认 64/128/256 主干处理完整 5000 点，输出 11 导联速度和独立 I 重建头。
- FiLM 恒等初始化、字段缺失、人口学编码、坏目标导联不进入 FM 状态。
- 线性路径端点、Euler/Heun 解析例、噪声不依赖 batch 排序、多样本 ODE 生成。
- 损失只做导数检查，未创建优化器或更新参数；CLI 无执行开关时明确拒绝训练。
- 安全 checkpoint/scales 校验及 Python/NumPy/Torch 随机状态序列恢复。
- 真正 WFDB 格式夹具，包含 ADC baseline/gain、mV→μV、aVR/aVL 排列与患者 fold 隔离。
- 实际已下载的 PTB-XL 第一条训练记录只读加载成功，形状 [12,5000]，物理电压有限。
- 独立 predict CLI 用临时随机权重、仅可见 I 运行成功；5017 点尾段保持，输出 [12,5017] 和 JSON sidecar。
- 原始目标/anchor 不减 median、固定尺度还原一次；当前 main 的 Task2 bonus 使用 V1–V6。
- 真实华为缓存与固定患者划分检查：938 个 train-only 同步窗、78 人；Task1 validation 231 窗/21 对，Task2 validation 300 窗/25 对。
- 将真实 validation target 自身作为评分输入时，两任务 r_missing11=1、RMSE=0；这是 evaluator 契约检查，不是模型成绩。

## 数据预检状态

华为缓存可用。公开下载正在进行，检查时 records500 仍有缺失；preflight 按预期返回 ready=false，未自动下载，也未启动训练。具体缺失数量会随下载变化，不作为固定统计写入源码。

## 验证环境

隔离的 CPU 测试依赖位于备赛工作区，不是本机全局训练环境：

- Python 3.12.14
- torch 2.14.1+cpu
- NumPy 2.5.3
- PyYAML 6.0.3
- wfdb 4.3.1

## 未执行及功能边界

没有启动公开预训练、华为微调或任何完整训练循环，没有训练模型 checkpoint，没有改写原始 Data/缓存。未进行 GPU 训练验证，也没有实际模型准确性结果。

首版只实现 B5-U 基础条件。历史 ECG/PPG、CNN＋Transformer 和正式主办方可执行包装仍属于后续独立扩展，不在本次首版中假装已实现。

未来训练环境请先安装 baselines/B5/requirements.txt，并按 README 配置数据、拟合/复用冻结尺度、完成 public preflight。实际训练需要显式 --execute-training。
