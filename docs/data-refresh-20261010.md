# 2026-10-10新数据处理结果

已刷新三个信号目录与userinfobean.csv，重建两任务独立context缓存和strict索引。原始Data文件未改写，旧缓存保留；未启动训练，未生成模型checkpoint。

## 原始数据与患者划分

| 来源 | 文件/行数 |
|---|---:|
| 心电图机d12 XML | 200 |
| 心电图机d6 XML | 200 |
| 体脂秤CSV | 134 |
| userinfobean.csv | 185行 |

旧110位患者的训练/验证归属全部保留。新增93位可靠HNU患者采用seed=42分配，划分表共203位：162 train、41 validation。划分表包含少量仅有旧记录/不可靠ID的患者，不等于可用于训练的d12患者数。

Task1按唯一准确患者ID配对，共126个paired、4个review；Task2共334个paired。没有对不可靠ID猜测匹配。人口学有6行无法可靠规范为HNU，原样保留以供复核；这不会删除可用ECG。

## 新版缓存

路径由configs/common.yaml指定：

- `../task1_record_context_v3`
- `../task2_record_context_v3`

| 数据 | 缓存target窗 | 质量筛选后训练窗/完整验证窗 | 配对数 |
|---|---:|---:|---:|
| Task1 train | 1188 | 1152 | 96个可训练配对 |
| Task1 validation | 324 | 324 | 27 |
| Task2 train | 3228 | 3156 | 263个可训练配对 |
| Task2 validation | 780 | 780 | 65 |

Task2可训练配对为156个machine d6与107个body-scale；验证为40个machine d6与25个body-scale。评分仍按设备1:1，不能因为40/25数量不同而按记录数混合。

每个target配对都使用真实120秒、12窗，不再受context长度限制。Task1 train/validation的独立context窗分别为1066/264；Task2为3558/859。context保留全时长、尾窗、采样mask、物理有效长度和只由有效点计算的记录median。

原始target/anchor不减median，context在模型batch中记录中心化；所有raw缓存为μV。padding在raw空间保存为context median，进入模型空间后为0，mask明确排除。旧数据未被填成假target。

## strict索引

`metadata/d12_strict_pretrain_index.csv` 已重建：1872个唯一训练窗口，156条d12记录/156位患者，每条12窗。索引来源Task1 1008窗、Task2 864窗；这是去重后的读取来源，不是两任务原始数据总量。

所有strict条目来自train患者；按target物理记录/窗口去重。reader检查array_index、record ID与物理起点是否对应新缓存，防止误用旧索引。

## 质量与回退

400份机器XML质量表已更新：200份d12中50条含质量警告，3条可靠导联不足6；200份d6中12条观察导联质量不合格。

新增`anchor_i_training_eligible`字段。2条d12虽满足可靠导联数量，但I被标记质量差；训练需要可靠同步I，因此P0和joint训练也排除这些记录。验证仍保留，raw数组不删除。其余部分坏导联继续仅排除对应Huber/PCC监督。

Task1 train/validation分别有9/5个配对回退anchor-only；Task2分别有11/1个machine d6配对禁用context。无效context不删除对应可用target。context编码与实际回退由模型分支使用mask/available字段实现。

## 人口学编码

原始userinfobean.csv为GB18030。`metadata/userinfobean_utf8.csv`已生成，与原始185行内容相同，未改原文件；该派生文件不进入Git。common.yaml提供demographics_csv路径，模型分支应使用它或main的兼容CSV reader。重复人口学记录仍保留，由模型adapter按既有规则处理冲突，不能随意选一条覆盖。

## 验证结果与备份

- 534份更新信号文件的SHA256复核通过，证明构建没有改写原始信号。
- 原始人口学CSV与UTF-8副本逐字段一致。
- 旧患者划分全部保持，所有缓存患者与权威划分相符。
- target有限且每个配对12窗/120秒；context median排除缺口和padding，batch无效位置为0。
- 两任务validation恒等预测r_missing11=1，RMSE=0；Task2设备加权加分=10。
- strict去重、train-only与可靠I检查通过。
- 合成mask/回退测试、逐记录设备评分、预处理及实验协议检查通过。

机器可读报告：`metadata/data_refresh_audit.json`、`reports/refreshed_data_checks.json`。刷新前的metadata备份在`reports/data_backups/refresh_20261010_232303/`，含旧strict索引与旧划分；旧task1_output_v2/task2_output仍在原处。

## 训练前的衔接

模型分支需要合并新Dataset/collator、设备metadata和四参数计分接口。main的数据重建不等于网络已经支持独立context。

从零训练可按新版train-only数据准备固定scale。若继续微调已有公开checkpoint，应沿用其兼容的冻结scale，不能直接替换为另一组scale。此次没有拟合模型训练scale，也未启动公开预训练或华为微调。
