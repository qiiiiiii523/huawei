# B5 诊断流程：不训练、不改 best.pt、不修改原始目标

本工具加入 baseline/B5，保持原有网络、训练、尺度和选模规则。best.pt 仍按原来的 Task1 相关性选择，last.pt 仍用于续训；不增加 best_score.pt。

两项工具都在 CPU 运行，使用已经存在的缓存或评估结果，不需要 GPU，不调用 train/validate 推理，不下载数据。新增依赖没有超出 B5 原 requirements：NumPy、PyYAML、SciPy。

## 1. 同步和测试

本机将新增文件提交推送后，服务器：

```bash
cd /home/qht/huawei
git status --short --branch
git fetch origin
git switch baseline/B5
git pull --ff-only origin baseline/B5
python -m unittest baselines.B5.tests.test_diagnostics -v
```

使用现有 BioFlow 环境即可。服务器有自己的配置修改时保留并处理，不执行强制reset。Data/data大小写按服务器实际配置保留，本次不修改数据路径配置。

## 2. 完整验证集快慢变化诊断

不重新生成预测；直接读取三组已保存的 prediction_uV.npy、target_uV.npy 和 window_metadata.csv：

```bash
python -m baselines.B5.diagnose_evaluation \
  --evaluation E1=results/B5/E1_local_meta/best_evaluation \
  --evaluation Public=results/B5/E2_public_meta/huawei_evaluation \
  --evaluation Finetune=results/B5/E2_finetune_meta/huawei_evaluation \
  --slow-seconds 1 \
  --output-dir results/B5/diagnostics/full_validation
```

如实际评估目录改过名字，把右侧路径改成包含task1/task2子目录的真实目录。各评估目录必须有数组，只有CSV报告不够。输出必须新建或为空；重复运行换目录名。

可选增加第35轮已保存的预测（不是恢复该轮模型）：

```bash
python -m baselines.B5.diagnose_evaluation \
  --evaluation E1=results/B5/E1_local_meta/best_evaluation \
  --evaluation Public=results/B5/E2_public_meta/huawei_evaluation \
  --evaluation Finetune=results/B5/E2_finetune_meta/huawei_evaluation \
  --evaluation Finetune35=results/B5/E2_finetune_meta/validation/epoch_0035 \
  --slow-seconds 1 \
  --output-dir results/B5/diagnostics/full_validation_with35
```

输出：

- report.md：三组完整Task1/Task2的原始r、慢变化r、快变化r和胸导联误差。
- task_summary.csv：可比较汇总。
- 每个模型/任务的 lead_diagnostics.csv：按导联汇总。
- record_lead_diagnostics.csv：每条完整评分记录的偏移、幅值及快慢误差。
- manifest.json：目标一致性、原始评分复算检查、滤波定义。

先按时间拼接完整评分pair，再在记录边界edge padding，使用501点（约1秒）移动平均分离慢变化。与之前抽样不同，不在每个10秒窗口边界重新滤波。r按记录平均，RMSE按点加权，与main聚合一致。记录覆盖不完整、重复窗口、非有限值或三组目标不同会报错。

原始r/RMSE复算会与已有overall_metrics.csv核对。滤波r和去均值RMSE都是诊断指标，不能当作正式成绩，也不能用真实目标的慢曲线或均值校正推理结果。快慢误差有交叉项，不能直接把两者MSE相加或当作互补百分比。

## 3. 原始XML到缓存的电压核对

```bash
python -m baselines.B5.audit_raw_voltage \
  --config configs/experiments/b5_local_meta.yaml \
  --max-records 5 \
  --output-dir results/B5/diagnostics/raw_voltage
```

默认从配置的data_root找原始文件。只需要心电图机d12 XML，不需要手表/PPG/体脂秤原始文件。自动选择验证集中V3均值绝对值最大的5条唯一目标记录，逐窗口核对两任务缓存。同时导出严格华为训练集和验证缓存的描述统计；不拟合尺度或校正器。

原始XML不在配置Data目录时明确指定：

```bash
python -m baselines.B5.audit_raw_voltage \
  --config configs/experiments/b5_local_meta.yaml \
  --raw-root /home/qht/huawei/data \
  --max-records 5 \
  --output-dir results/B5/diagnostics/raw_voltage_retry
```

这里raw-root应是实际含“心电图机d12”子目录的目录。把示例的小写data换成实际目录；工具兼容元信息里Data/...前缀和Windows路径，但不擅自改变实际文件夹大小写。

服务器没有原始XML时：工具仍输出缓存分布和 required_raw_files.csv，退出码2表示核对未完成，不代表训练坏了。根据required_raw_files.csv从本机Data/心电图机d12上传所需XML，然后用新输出目录重跑，不必上传整个原始数据集。

也可明确指定此前困难记录：

```bash
python -m baselines.B5.audit_raw_voltage \
  --config configs/experiments/b5_local_meta.yaml \
  --record-id d12_b9e3e1526f5aac \
  --record-id d12_3fdafd49fbef07 \
  --record-id d12_0b2b97044d07ae \
  --output-dir results/B5/diagnostics/raw_voltage_known3
```

工具独立读取最长完整12导联XML序列，按各导联声明的digits、scale、origin和单位换算为μV，核对当前main解析器的输出，再复用同一重采样协议核对缓存。输出xml_calibration.csv和逐窗口cache_comparison.csv；audit.json中checks_complete_and_matching=true才表示选定记录已完成且匹配。

这能验证声明的转换与缓存一致，不能独立证明设备校准或每条信号的生理有效性。不能因目标偏移大就删除记录或改减median。

## 4. 看什么、发什么

```bash
cat results/B5/diagnostics/full_validation/report.md
cat results/B5/diagnostics/raw_voltage/audit.json
```

确认：

1. 原始r与旧报告一致；三组目标和清单一致。
2. full_validation中的完整记录快变化r是否普遍高于原始r，还是只有个别导联/记录提高。
3. XML转换与缓存是否匹配；训练/验证记录均值分布是否差异明显。

打包结果，无需上传模型或大波形：

```bash
tar -czf B5_followup_diagnostics.tar.gz -C results/B5 diagnostics
```

只打包diagnostics目录，不包括完整results模型/预测数组。将文件下载到本机后提供分析。

当前不实现新的慢变化网络/损失、不训练新模型。根据诊断结果再设计单因素实验，最终始终在原始μV下评估。

## 开发验证

新增测试覆盖：O(T)移动平均与卷积一致、常数偏移不改变r、快慢误差交叉项、常数信号未定义相关性、记录完整性、与main记录宏平均/点加权RMSE一致、CLI全记录处理和跨模型目标不一致拒绝、XML单位/origin/最长序列/1000→500Hz重采样、错误缓存偏移检测、跨平台路径与路径越界拒绝。

本机三条真实困难记录的XML与缓存核对已通过。本机没有服务器完整预测数组，所以真实完整验证集的快慢结果需在服务器运行；没有启动训练。
