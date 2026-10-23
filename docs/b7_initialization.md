# B7独立初始化：不继承B5

2026-10-24用户决定：取消B5权重迁移。该决定替代先前所有“默认部分迁移B5”的方案。

- `B7UNet`由B7自己的`ModelConfig`直接构建，首阶段使用固定seed的随机初始化。
- 无B5模型、配置、权重或迁移模块的导入，不需要原B5的.pt文件。
- `train.py`不传`--init-checkpoint`即为随机初始化；每个run记录`kind=random_B7`及`B5_weights_used=false`。
- 后续公开长窗、华为适配可以使用`--init-checkpoint`，但只接受架构ID为`B7-U7-raw-I-v1`、相同配置和scale的B7 checkpoint。B5包会明确报错，不部分加载。
- `--resume`只用于同一个B7 run的`last.pt`，校验源码、配置、预算、manifest及记录文件SHA，并恢复优化器、EMA和RNG。
- 新B7阶段从上一阶段的EMA开始，优化器/最佳记录重新建立；保留本阶段initial，允许第0步胜出。
- scale从B7公开训练划分拟合，不读取B5的scale；后续阶段通过`--scale-manifest`使用同一份冻结scale。

U7仍选用三尺度一维U-Net骨干。这是架构选择，不是继承旧参数。UT7、纯Transformer T7目前仍是方案候选，尚未实现，不能作为可运行模型宣传。
