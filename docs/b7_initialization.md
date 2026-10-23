# B7-T初始化：独立随机训练

2026-10-24。当前分支从B7提交`1615b21`创建，骨干由U7替换为时序Transformer。

- `B7Transformer`按本分支`ModelConfig`构建；首阶段不传`--init-checkpoint`，固定seed随机初始化。
- 不迁移B5或B7-U7权重；checkpoint架构ID为`B7-T-temporal-transformer-v1`，旧U7/B5在加载时明确拒绝，不使用strict=False凑合加载。
- 初始化记录`kind=random_B7T`、`B5_weights_used=false`、`B7_U7_weights_used=false`。
- 后续阶段只接受相同架构、配置和冻结scale的B7-T checkpoint，使用上一阶段EMA重新建立优化器与best记录。
- `--resume`仍只用于同一个run的`last.pt`，核对源码、配置/预算、manifest及记录SHA并恢复优化器、EMA和RNG。U7的训练目录不能用于T的resume。
- 原B7的原始波形缓存、患者split、QC和scale协议完全保留，因此符合协议的数据manifest可以复用。权重与数据缓存是两件事。
- 实验顺序仍是两个短窗公开集联合训练P_T，再分叉为华为A_T及PTBDB长窗Q_T，Q_T后再华为B_T。新架构不从原U7的P开始。

默认运行输出请放`results/B7-T/`，与U7的`results/B7/`区分。命令模块与配置文件名沿用`baselines.B7`及`configs/b7_*.json`，本分支的配置已全部指向Transformer。完整结构见[架构说明](b7_architecture.md)。
