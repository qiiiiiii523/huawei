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

七层需要至少64个native点：5Hz真实10秒只有50点，因此public10采用10Hz（100点）；不复制或拉伸真实10秒成120秒，不跨记录拼接。public75有375点，华为120秒有600点。10Hz→5Hz会改变卷积对应的物理时间范围，这是迁移方案的一部分；本方案不能视为仅改变参数量的严格单变量消融。严格结构消融需给对照模型相同的阶段分辨率、预算和优化设置。`build(..., input_seconds=75, output_seconds=75)`支持长记录训练；训练器的sample_window用于真实10秒/75秒最终验证，predict_slow保留120秒官方推理合同。

## 优化、损失与评估

参考M7：AdamW，lr=1e-4，weight_decay=.01，clip_grad_norm=1，EMA=.999，seed42；每500更新保存恢复点。使用真实batch64，不拆microbatch、不做梯度累积：一次完整前向/反向执行一次optimizer及EMA更新。已在物理GPU2以真实数据验证10/75/120秒三阶段batch64；120秒前向反向的PyTorch张量峰值约1.57GiB，不含其他进程、CUDA上下文及训练优化器状态。

损失先保留此分支LossConfig默认：七通道FM速度MSE + .1×MAE7 + .05×PCC7，II权重1；这是本次计划值，不声称从旧训练日志复原。MAE/PCC用于端点估计x_t+(1-t)v。训练标签为全记录500Hz下先做edge-padding MA501、再滤波降至native_hz的II和V1–V6；不能把原始全频标签当慢标签。mask无效位置应在算术前排除；人口学缺失编码沿用B5。I不加噪、不作为预测标签。训练noise_std7、scale须按原协议固定，任何重新估计只使用训练患者并记录新值，不能从验证集拟合。

没有直接复制M7完整波形的MAE=1、PCC=.1、II=5或LF附加项，因为当前目标是慢分量；若要调整损失，请作为独立实验，与相同网络、数据和预算的基础损失比较。

参考M7，每阶段只做最终EMA验证，避免频繁完整K16验证；本轮不搜索best-step。华为最终使用原可靠患者/QC/Task2设备等权规则，以固定快+新慢的完整120秒11导联r为主要结果，同时记录RMSE和快/慢诊断r。统一Heun16、K16正负噪声配对、稳定record_key，seed42，sample_batch16、record_batch4（显存不够仅缩小这两个执行批量，不降低K或步数）。若需要best-step选择，请另行明确验证频率并匹配对照。

## 实际训练入口

现已补齐 `FM_Unet.train`，直接读取不可变B7 raw-uV 500Hz缓存。训练前校验患者/目标跨split泄漏、QC-v2、源数据类型、长度与内容SHA；训练标签在完整真实记录上先MA501，再整理到50Hz并滤波降到目标分辨率，最后提取真实窗。按source→patient→record均衡抽样。按需生成native训练缓存，最多保留512条内存缓存，不重复解码全部公开波形。

归一化scale_uV及noise_std7只读取随附best.pt的冻结统计，不从其中加载任何网络权重，也不重新拟合验证数据。华为缺少缓存人口学字段时，从原userinfobean.csv用原B5编码及冲突处理规则补全；该CSV的SHA也绑定恢复身份。固定快模型使用原FM_P_best45 EMA，严格核验state SHA；K16正负配对、Heun16，先拼完整120秒再减MA501。快模型不加入训练优化器，且只在华为最终验证时加载。

```bash
# 已激活BioFlow，在本分支服务器工作目录执行；脚本固定物理GPU2。
bash scripts/run_fm_unet_m7.sh
```

脚本默认数据路径已在服务器核实，依次运行public10→public75→huawei120。第一阶段随机初始化，后续加载上一阶段final.pt的EMA并重置优化器。每阶段没有初始/中间验证，每500更新保存last.pt，每50更新打印损失；每阶段结束只验证一次最终EMA。公开阶段报告slow-only窗口诊断，华为报告fast+slow完整120秒主要评分，二者不混用。当前验证逐记录处理（record_batch实际上为1），每条记录轨迹batch按配置16执行，不改变K16结果。

输出位于 `results/FM_Unet_M7/seed42_时间戳/`，并在 `results/FM_Unet_M7/latest_run.txt` 记录目录。日志分别为logs/public10.log、public75.log、huawei120.log。每阶段保存last.pt、final.pt、inference.pt、validation_final.json及completed.json。`final.pt`含完整恢复信息；只有最终验证成功后才生成，用于下阶段初始化。`inference.pt`仅包含EMA、模型配置与冻结统计；正式120秒推理使用huawei120/inference.pt。

如中断，指定相同RUN重新执行脚本：

```bash
RUN=/原实验完整目录 bash scripts/run_fm_unet_m7.sh
```

脚本跳过已完成阶段，对有last.pt的当前阶段自动--resume。精确恢复严格绑定代码、完整配置、manifest/训练与验证波形SHA、归一化/噪声统计、人口学CSV、设备类型及快模型身份；变更这些内容需另开目录。恢复可重做未完成的最终验证，不重做已完成的优化器更新。测试已验证恢复后model/EMA与连续训练逐值一致。

手工单阶段命令（第一阶段，无旧网络初始化）：

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID python -u -m FM_Unet.train \
  --manifest /home/qht/huawei/data_cache/B7_public_qcv2/data/public10/manifest.json \
  --config configs/fm_unet_m7/public10.json \
  --output results/FM_Unet_M7/manual_public10 \
  --cache-dir results/FM_Unet_M7/manual_cache --device cuda:0
```

配置检查仍可用 `python -m FM_Unet.experiment --config configs/fm_unet_m7/huawei120.json`；它只检查计划，不是训练入口。不要运行M7的baselines.B7.pretrain来替代本慢FM训练器。验证：`python -m unittest discover -s FM_Unet/tests -v`。
