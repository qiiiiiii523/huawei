# B7-T：一维多尺度Transformer条件Flow Matching

2026-10-24。从B7@1615b21创建独立B7-T分支，原U7保留在B7分支。本分支只替换骨干：20毫秒联合patch→局部平移窗口Transformer→全局瓶颈→跳连解码→完整七通道速度。三尺度128/256/384，12块，参数15,063,342。没有Conv1d骨干，没有独立I编码器。

包名仍为`baselines.B7`，配置名仍为`configs/b7_*.json`。不传初始化参数即从零训练T，后续只接受同架构B7-T checkpoint；B5/U7权重明确拒绝。B7缓存和冻结scale可以复用，结果写入`results/B7-T/`，不覆盖原U7结果。数据、loss、采样、评分及优化预算与B7相同。

## 实现合同

- 每个窗口完整I直接作为第一个输入通道，不先经独立I编码、不加噪、不做ODE更新。
- 网络输入8通道(I+7状态)，输出7个速度，联合生成II与V1～V6；I直接复制，四条派生肢体导联在μV下计算。
- 40秒输入输出均20000点；FM/MAE覆盖全窗有效位置，PCC固定只算中央7500:12500点(15～25秒)。公开10秒时PCC算完整真实10秒。
- 总loss为FM MSE+1.0×MAE+0.1×PCC，导联权重(5,1,1,1,1,1,1)按有效权重归一化。
- Heun16、K16平均；40秒滑窗每次取中央10秒、步长10秒拼完整记录。首尾重复端点延拓，人工位置无监督。
- 华为验证要求真实完整120秒及task1/machine/body三组，按0.5/0.25/0.25聚合全部11条缺失导联的完整记录r选best_r。

## 环境与首阶段

自动调度入口及数据准备恢复已同步源B7@0113fbc：`python -m baselines.B7.pipeline --data-root /home/qht/huawei/data --metadata /home/qht/huawei/metadata --ptb-root /home/qht/huawei/data/PTB-Diagnostic`。默认输出`results/B7-T/auto_seed42`，支持P结束后并行A/Q、Q结束后立即B、失败阻断依赖、断点恢复及分数/权重路径汇总。支持Chapman粘行头及训练开始前显式`--retry-preparation`：逐条对照原始数据核验并继续部分缓存。完整启动命令见[服务器说明](../../docs/b7_server_data.md)。这是可运行入口，尚未在服务器启动真实T训练。

从仓库根目录运行。依赖：`python -m pip install -r baselines/B7/requirements.txt`。下面命令只有显式调用train才训练；本轮未执行真实ECG训练。

先把两个短窗公开集一起准备为B7缓存，自动使用PTB-XL的patient/fold和Chapman按患者固定划分，scale只从train拟合：

```powershell
python -m baselines.B7.prepare --public-source public10 --root 'C:/Users/Ashley/Desktop/HW/Data' --output results/B7/data_public10
python -m baselines.B7.train --manifest results/B7/data_public10/manifest.json --config configs/b7_public10.json --output results/B7-T/public10_seed42 --device cuda
```

第一条会读取较多原始文件并写新缓存。默认公开QC只保证有限数据，不替代已审计的信号质量筛选；有既定QC/split时用显式index。不会改写原始数据或将不同患者10秒拼成40秒。

## 两条后续数据路线

服务器已有元数据可通过新入口`python -m baselines.B7.huawei_index`转换成显式index，详见[服务器数据准备](../../docs/b7_server_data.md)。该入口明确沿用B5的同步d12-I监督，不把独立手表/d6 context作为逐点对齐的I；本入口尚未映射人口学表，华为人口学按未知处理。

A先做：同一公开10秒候选P直接华为40秒。

B后做：从同一P进行PTBDB40秒，再华为40秒。PTBDB按WFDB头的文件映射读取标准12导联(.dat)，额外Frank3在.xyz；它不是异常头文件。

```powershell
python -m baselines.B7.prepare --public-source ptbdb --root 'C:/Users/Ashley/Desktop/HW/Data/PTB_db' --scale-manifest results/B7/data_public10/manifest.json --output results/B7/data_ptb40
python -m baselines.B7.train --manifest results/B7/data_ptb40/manifest.json --config configs/b7_public40.json --init-checkpoint results/B7-T/public10_seed42/best_r.pt --output results/B7-T/public40_seed42 --device cuda
```

华为缓存必须来自既定患者划分、可靠I与逐导联QC，不能自动将全部原始文件当训练集。准备入口支持显式index：

```powershell
python -m baselines.B7.prepare --index huawei_index.json --scale-manifest results/B7/data_public10/manifest.json --output results/B7/data_huawei
python -m baselines.B7.train --manifest results/B7/data_huawei/manifest.json --config configs/b7_huawei40.json --init-checkpoint results/B7-T/public10_seed42/best_r.pt --output results/B7-T/huawei_A_seed42 --device cuda
python -m baselines.B7.train --manifest results/B7/data_huawei/manifest.json --config configs/b7_huawei40.json --init-checkpoint results/B7-T/public40_seed42/best_r.pt --output results/B7-T/huawei_B_seed42 --device cuda
```

配置给出首轮起始预算(公开10秒10000更新，后续各1000更新)，未通过真实训练优化。新阶段新目录；同阶段恢复在原命令加`--resume`并移除`--init-checkpoint`。源码/预算/数据变化时拒绝精确resume。best_r含initial候选。

`best.pt`与`best_r.pt`是相同的验证r最佳候选，两者均保存EMA；不按训练窗口r或总loss选best。日志中的`train.window_r`仅作训练监测。最终评分逐导联先平均同目标配对、再平均目标、最后平均11导联，保持原公共评分口径；未定义Pearson沿用原规则处理。

## 显式index合同

JSON顶层为记录列表，每条有下列字段：

```json
[
  {
    "record_id": "task1:pair1",
    "target_record_id": "d12:record1",
    "subject_id": "huawei:patient1",
    "source": "huawei",
    "group": "task1",
    "split": "train",
    "input_reliable": true,
    "path": "raw_pair1.npz",
    "lead_mask": [true,true,true,true,true,true,true,true,true,true,true,true]
  }
]
```

`path`相对index文件，支持.hea、完整d12.xml、raw-uV.npz。NPZ内`target_uV[12,T]`，可选`anchor_uV[1,T]`和`sample_rate`(默认500)；anchor未提供时使用同步目标I。手表等可见I应使用真实对齐的anchor，不能用目标I替代后声称完成真实输入评估。group取task1/machine/body，split取train/validation/test。需要三组验证记录；同一物理target跨配对保持同样target_record_id与QC。

XML读取最长完整12导联并重采样；华为取前120秒，同步配对的物理起点若不同应先生成对齐NPZ。NPZ/公开缓存不预先编码I，只保存原μV。可选`metadata`对象保存已编码numeric(3值)、sex(0男/1女/2未知)、field_mask(4值)、age_topcoded(1值)；不完整字段会报错，完全缺失用未知编码。

## 验证、导出与推理

```powershell
python -m baselines.B7.evaluate --manifest results/B7/data_huawei/manifest.json --checkpoint results/B7-T/huawei_A_seed42/best_r.pt --output results/B7-T/huawei_A_evaluation.json --device cuda
python -m baselines.B7.package --checkpoint results/B7-T/huawei_A_seed42/best_r.pt --output results/B7-T/B7_T_inference.pt
python -m baselines.B7.inference --checkpoint results/B7-T/B7_T_inference.pt --anchor current_I_uV_500Hz.npy --record-key stable_record1 --output results/B7-T/prediction.npy --device cuda
```

推理只接受可见I与可选人口学，不接受隐藏目标；输出[12,T]、单位μV，T必须是完整10秒的整数倍。正式华为为60000点。公开验证只使用实际完整10秒块；不足一个块的尾部不补成标签。训练窗口仍可从完整真实长度随机抽取。

## 检查与范围

```powershell
python -m unittest discover -s baselines/B7/tests -v
```

58项测试通过：原B7的24项核心行为、5项华为桥接检查、10项自动调度检查、4项数据准备恢复检查，以及15项Transformer检查（全采样分片、局部边界、全局长程依赖、变长/补齐、输出顺序、条件注入及U7权重拒绝）。额外使用正式15M参数模型完成40秒合成CPU反向传播和AdamW更新，全部参数梯度有限。完整120秒Heun16/K16的控制流沿用轻量速度场检查，不表示已完成正式网络GPU全量验证。

本分支已实现T；原U7仅保留在B7分支，UT7仍未实现。尚未进行真实训练或GPU验证，不保证T优于U7。详见[当前验证记录](../../docs/b7_t_validation.md)。

文档：[完整方案](../../docs/b7_plan.md)、[逐层结构](../../docs/b7_architecture.md)、[从零初始化](../../docs/b7_initialization.md)。
