# B7-T实现与验证记录

日期：2026-10-24。来源：B7提交`1615b21`，独立分支/工作树`B7-T`。测试范围为合成CPU行为和架构规则，不是模型医学性能或GPU基准。

## 架构已替换

`B7Transformer`取代`B7UNet`，独立架构ID为`B7-T-temporal-transformer-v1`。10点联合patch、128/256/384维、2/2/4编码深度、2/2解码深度；细层/中层平移窗口，粗层全局注意力，线性逐点恢复七个速度。共15,063,342参数，无Conv1d骨干。

完整40秒得到2000→1000→500→1000→2000个token，最终输出[1,7,20000]；10秒使用同一参数得到500→250→125→250→500个token。没有独立I编码分支；I原始通道先与状态拼接，再进入共享patch映射。

## 回归与新增检查

```bash
python -m unittest discover -s baselines/B7/tests -v
```

共42项：继承B7的24项核心行为（模型实例替换为T）、5项华为索引桥接，以及13项Transformer专门检查。全部通过。

- 原B7行为：40秒输出形状、直接I/不加噪、7维flow、整窗MAE/中央PCC、II权重、QC/mask、真实窗口、K16平均、拼接/肢体公式、评分器一致性、数据划分/scale冻结、checkpoint导出和精确训练恢复。
- Transformer：patch包含完整采样且时间顺序正确；输出恢复采样/导联顺序；局部平移连接相邻窗口但不首尾循环；padding key被屏蔽且全空窗口有限；奇数token和短片段恢复原长度；40秒远端I可影响中央预测；时间及人口学条件有效；短长窗位置前缀一致；没有卷积骨干；U7权重明确拒绝。
- 实际Transformer的Heun调用中，每次共享patch入口的I逐点保持相同。
- 完整120秒、Heun16/K16控制流依旧用轻量速度场检查6144次调用，不将该检查说成正式Transformer全量生成或GPU性能验证。

## 正式宽度40秒反向传播

除小模型回归外，正式15M参数配置在CPU上对合成[1,7,20000]标签完成一轮完整FM+MAE+PCC反向传播、梯度裁剪和AdamW更新。所有可训练参数均有有限梯度，全局注意力QKV梯度非零，I未被修改，更新后参数全部有限。详细环境和数值见[b7_t_cpu_probe.json](b7_t_cpu_probe.json)。单次合成损失及耗时均不代表真实训练成绩或服务器吞吐。

## “其他规则不动”的对照证据

相对源提交逐文件核查20个文件，包含数据读取/准备/华为桥接、flow、loss、肢体公式、推理、B7评分、原公共评分器、导出、依赖及metadata，内容完全相同（仅统一CRLF/LF后比较）。

三份配置仅`model`字段不同，loss、sampling、training与stage逐项相同。LossConfig、SamplingConfig、TimeEmbedding及MetadataEncoder的AST与源B7相同。训练脚本去除模型类名及初始化来源标记的必要差异后，与源B7完全相同；未改AdamW/EMA、预算、采样概率或best选取逻辑。原噪声key协议也保留，以便架构比较使用相同种子条件。

详细SHA与配置对照见[b7_t_scope_audit.json](b7_t_scope_audit.json)。

## 使用边界

新T不兼容原U7/B5权重，需要从随机T重新训练P_T；后续仅接受相同配置的T checkpoint。已准备的B7原始波形manifest、固定患者划分/QC及冻结scale可以复用，结果使用独立目录`results/B7-T/`。CLI包名保持`baselines.B7`，三份`configs/b7_*.json`已指向T。

尚未进行真实公开预训练、华为微调、GPU显存/速度或官方完整输入评估；没有T优于U7的性能结论。原B7分支保持U7，UT7混合模型未实现。
