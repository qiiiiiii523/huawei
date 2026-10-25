# 七层慢FM：M7宽度实验计划

## 模型与固定快分支

默认 `build()`：七个分辨率层级，通道64/128/256/256/256/256/256，时间/人口学嵌入128维；每层编码、解码各2残差块，最底层6块（d=1/2/4/8/16/32）。参数实测14,506,847，指慢模型本身，不包含固定快模型。M7-L是14,502,815；多出的4,032来自stem由8通道增加到17通道（原始I每10点打包成10通道，再拼接7个生成状态），9×64×7=4,032。

快模型及权重不变，只训练慢模型参数；不把快模型加入优化器。最终仍然fast11+slow11，不叠加旧慢CNN、G120或0.75系数。旧best.pt仍是三层最佳EMA，不能作为新宽七层权重或训练恢复点。

## 主实验预算

| 配置 | 数据 | 原始I | 慢状态Hz | 实际输入/输出秒数 | 更新次数 | batch |
|---|---|---|---|---|---|---|
| public10.json | PTB-XL + Chapman真实短记录 | 500Hz | 10 | 10/10 | 2,000 | 64 |
| public75.json | PTB Diagnostic真实长记录 | 500Hz | 5 | 75/75 | 4,000 | 64 |
| huawei120.json | 华为原患者划分 | 500Hz | 5 | 120/120 | 6,000 | 64 |

三阶段串行：第一阶段随机初始化，后续阶段接上一阶段EMA作为模型起点，优化器重新开始；这属于阶段迁移，不是中断续训。中断恢复则应恢复同一阶段模型、EMA、优化器、step和全部随机状态。相邻阶段模型配置完全相同，不按native_hz重新缩放瓶颈dilation。

此预算沿用原慢实验2000/4000/6000，先评估加宽加深后的整体表现。M7 public10原配置是20,000更新；若采用该预算，请另开实验输出目录，仅把public10的updates改为20000，勿将额外训练收益归因为结构收益。

七层需要至少64个native点：5Hz真实10秒只有50点，因此public10采用10Hz（100点）；不复制或拉伸真实10秒成120秒，不跨记录拼接。public75有375点，华为120秒有600点。10Hz→5Hz会改变卷积对应的物理时间范围，这是迁移方案的一部分；本方案不能视为仅改变参数量的严格单变量消融。严格结构消融需给对照模型相同的阶段分辨率、预算和优化设置。`build(..., input_seconds=75, output_seconds=75)`支持长记录训练前向，但现有predict_slow仍仅用于120秒官方推理，不能用它直接验证75秒记录。

## 优化、损失与评估

参考M7：AdamW，lr=1e-4，weight_decay=.01，clip_grad_norm=1，EMA=.999，seed42；每500更新保存恢复点。batch64是计划有效批量，GPU显存未实测；如需microbatch16累积4次，损失除4且只在一次有效更新结束后执行optimizer/EMA/step，更新总预算保持不变。

损失先保留此分支LossConfig默认：七通道FM速度MSE + .1×MAE7 + .05×PCC7，II权重1；这是本次计划值，不声称从旧训练日志复原。MAE/PCC用于端点估计x_t+(1-t)v。训练标签为全记录500Hz下先做edge-padding MA501、再滤波降至native_hz的II和V1–V6；不能把原始全频标签当慢标签。mask无效位置应在算术前排除；人口学缺失编码沿用B5。I不加噪、不作为预测标签。训练noise_std7、scale须按原协议固定，任何重新估计只使用训练患者并记录新值，不能从验证集拟合。

没有直接复制M7完整波形的MAE=1、PCC=.1、II=5或LF附加项，因为当前目标是慢分量；若要调整损失，请作为独立实验，与相同网络、数据和预算的基础损失比较。

参考M7，每阶段只做最终EMA验证，避免频繁完整K16验证；本轮不搜索best-step。华为最终使用原可靠患者/QC/Task2设备等权规则，以固定快+新慢的完整120秒11导联r为主要结果，同时记录RMSE和快/慢诊断r。统一Heun16、K16正负噪声配对、稳定record_key，seed42，sample_batch16、record_batch4（显存不够仅缩小这两个执行批量，不降低K或步数）。若需要best-step选择，请另行明确验证频率并匹配对照。

## 如何使用配置

配置文件是实验计划；当前分支没有完整慢FM数据加载、训练、EMA更新、断点恢复及分阶段验证入口。不要直接运行M7的baselines.B7.pretrain：它训练全频B7，不会自动应用本慢分支的MA501、native分辨率或固定快相加评估。

```bash
python -m FM_Unet.experiment --config configs/fm_unet_m7/huawei120.json
python -m unittest discover -s FM_Unet/tests -v
```

第一条仅检查配置/实例化网络并输出参数量，不启动训练。已有训练器可通过下面接口读取计划；仍须由训练器实现上面的目标、优化更新和验证逻辑：

```python
from FM_Unet.experiment import load_experiment, build_optimizer
cfg, model = load_experiment("configs/fm_unet_m7/huawei120.json")
model = model.to("cuda")
optimizer = build_optimizer(model, cfg)
```

训练检查点必须保存model_config、native_hz、input_seconds、output_seconds、ema、scale_uV、noise_std7，以及恢复用optimizer/step/RNG。输出目录与旧三层/窄七层实验分开。当前只完成模型、配置及CPU验证，没有训练成绩或GPU显存结论。
