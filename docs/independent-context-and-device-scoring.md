# 独立记录context与Task 2设备等权评分

更新日期：2026-10-10。本次只修改main代码和配置，未复制、删除、解析或重切真实数据，未训练模型。

## 1. 数据关系

```text
跨时间context完整记录 → 有效点median → 中心化和设备scale → 独立10秒窗＋尾窗＋mask
                                                            ↓
                                      模型分支：编码各窗、mask汇总为记录条件
                                                            ↓
同步I＋target同一120秒 → 固定12窗 → anchor编码、融合 → 12个target预测窗
```

context第k窗与target第k窗没有同步含义。context窗口数可以少于或多于12；每个target窗口关联同一条context的全体窗口，不用target序号索引context。context-target逐点损失禁止。

## 2. main公共接口

- `window_target_record(raw_d12_uV, start_sample=0)`：取真实120秒，生成 `[12,12,5000]`。短target报错，禁止补target；构建脚本记录并排除不足120秒的记录。
- `window_context_record(raw_context_uV, source, valid_mask)`：按实际长度独立切窗，保留尾段，返回 `ContextWindows`。
- `IndependentRecordCacheBuilder`：两个任务通用，保存独立target/context索引，不建立同序号窗对应关系。
- `JointAnchorDataset`：新版默认Dataset，返回 `RecordContextSample`；`context_ecg` 是 `[W,C,5000]`。
- `collate_record_context`：NumPy batch、统一变换、窗口数量padding、mask、有效时长权重和可选训练context dropout。转Torch、编码和融合由模型分支完成。
- `prepare_record_context_inference`：只接受可见同步I、context、mask和冻结preprocessor，没有target参数。I必须有真实120秒。
- `LegacyJointAnchorDataset`：仅供明确的旧缓存审计，不能生成丢失的context尾段或target窗口，不用于新版流程。

### 中心化与mask

context median从原始有效点逐导联计算，排除缺口、NaN/Inf和padding。原始空间缺口/尾窗padding保存为该导联median，所以模型空间填充值为0。有效性由mask表达，不根据“数值是否为0”判断。

batch字段：

| 字段 | 形状/含义 |
|---|---|
| context | `[B,W_max,C_max,5000]`，中心化且缩放后；无效位置为0 |
| context_time_mask | 同形状，排除缺口、尾段padding、坏导联和padding窗口 |
| context_window_mask | `[B,W_max]`，至少有一个有效采样点才为true |
| context_valid_lengths | `[B,W_max]`，padding前的物理窗长；不等同于有效点数量 |
| context_valid_counts | `[B,W_max]`，用于汇总的有效导联采样点总数 |
| context_window_weights | 有效点数量归一化；缺口不计权重，尾窗按有效时长贡献 |
| context_available | `[B]`，false时模型必须回退anchor-only |
| anchor_i / target | 原始电压只除固定scale，不减median |
| target_quality_mask | 12导联监督mask，传公共loss |
| evaluation_metadata | 包含pair ID、target record ID、窗口起点、expected_window_count和input_type |

缺失、全无效或QC不合格的context不删除有效target。machine d6明确质量不合格时整个context禁用。训练dropout需显式指定概率和有种子的NumPy generator；验证不能随机dropout。

模型分支必须在编码与汇总中实际使用mask。不能把padding窗口作为正常token，也不能对全padding记录直接做未保护的softmax。main只提供数据、权重和回退标记，不实现网络pooling/fusion。

## 3. 独立缓存格式

每个任务目录包含：

```text
taskX_train_target.npy                     [N_target,12,5000]
taskX_validation_target.npy
taskX_train_context.npy                    [N_context,C,5000]
taskX_validation_context.npy
taskX_train_context_valid_mask.npy          [N_context,C,5000]
taskX_validation_context_valid_mask.npy
taskX_window_metadata.csv                   target索引：pair、target位置、input_record_id、input_type
taskX_context_window_metadata.csv           context索引：物理记录、位置、有效长度、记录median
taskX_build_audit.json                       independent-record-context-v1
```

两个数组的N不需要相等。context记录按来源/record ID复用一次；pair负责将context物理记录与target关联。不同pair不能跨患者/划分。target缓存保留原始μV，不写去median后的label。

## 4. 构建代码已准备，但本次没有执行

common.yaml已指向尚未生成的 `../task1_record_context_v3` 和 `../task2_record_context_v3`。旧task1_output_v2/task2_output不动，也不自动迁移。新数据扫描、配对、QC和患者归属仍待后续数据处理任务完成。

完成metadata审核后，后续可显式运行：

```powershell
python scripts/build_task1_record_windows.py --data-root .. --output-dir ../task1_record_context_v3
python scripts/build_task2_record_windows.py --data-root .. --output-dir ../task2_record_context_v3
python scripts/build_d12_strict_pretrain_index.py
python scripts/check_task1_record_cache.py
```

两构建脚本均拒绝覆盖已有输出目录。Task1不再取watch/target最短长度，也不再硬编码104条/83:21数量。Task2保持canonical d6；体脂秤reader按CSV单位转换μV，保留Index内部缺口并标无效，不将非零初始Index当缺口。

所有target从选定真实120秒区间切窗；同步I从该区间target I得到。context保留其原始有效完整时长。跨时间记录不做窗口级硬对齐。构建脚本默认target区间从0开始；如接口明确其他校准起点，通用writer支持显式target_start_sample。

重建缓存会改变array_index，必须重建strict索引。P0 reader检查新缓存中的record ID/物理起点，拒绝把旧索引套在新数组上。scale仍仅训练拟合并冻结；数据处理后的正式新训练需重新准备scale，旧checkpoint不能随意更换scale。

## 5. Task2评分

每个pair严格12窗/120秒，按窗口起点拼接，再算II–V6逐导联r及记录平均。Task2 metadata必须有input_type，两设备均须存在。

```text
machine_r = 心电图机d6组逐记录r_missing11
body_r    = 体脂秤组逐记录r_missing11
Task2 r   = 0.5 × machine_r + 0.5 × body_r

machine_RMSE / body_RMSE：分别计算V1–V6每导联采样点加权RMSE，再平均六导联
b(x) = 10（x ≤ 70 μV）；否则700/x
Task2加分 = 0.5 × b(machine_RMSE) + 0.5 × b(body_RMSE)
```

设备组内记录平均方式沿用main；设备之间严格1:1，不按人数、记录数或窗口数赋权。两设备的RMSE均值只作诊断，不再拿这个均值换算加分。0/140 μV例子应得到7.5分，而不是10分。

summary包含两个设备的r、胸导联RMSE、加分和记录数，以及等权后的r_missing11、task2_rmse_bonus_score与协议标记。CSV总分脚本拒绝旧的混合设备报告。

`competition_score(r1, r2, machine_rmse_uV, body_rmse_uV)` 已改为四参数；r2传设备等权结果。旧三参数调用必须修改。主分仍为0.5×r1+0.5×r2，总分加上设备等权RMSE加分。

评估只接收还原后的μV，不再次减median、乘scale、滤波或替换I。

## 6. 模型分支需要改的调用

1. Dataset从一张context窗变为全记录窗口集合，使用新版collator。
2. 编码/融合网络实现记录条件mask汇总，并在context_available=false时回退anchor-only。
3. validation保存input_type与expected_window_count；不用context窗口序号充当target时间位置。
4. 用新版四参数competition_score或公共汇总脚本，禁止用平均RMSE换加分。
5. 严格按120秒target记录评分；110秒旧Task1缓存会报错，不能填target补分。

本次没有修改任何模型分支，也未验证网络精度或启动训练。合成临时缓存覆盖mask、尾窗、有效median、完整target、Dataset关联、dropout、回退及设备评分测试。
