# B5 同步输入 I、真实胸导联、预测胸导联慢曲线对照

在 `baseline/B5` 使用独立只读工具 `baselines.B5.plot_anchor_slow`，不训练、不推理、不加载 checkpoint、不改变正式评估或 best.pt。

## 数据来源与核对

- 预测和真实目标：指定评估目录的 `task2/prediction_uV.npy`、`target_uV.npy`、`window_metadata.csv`。
- 输入 I：同一配置下 `JointAnchorDataset.anchor_i_ecg`，即 B5 实际使用的同步心电图机 I；不是预测数组里的 I，也不是异步手表/体脂秤 context。
- 当前验证协议用目标中的 I 模拟测试时可见的同步 I。本工具读取这个正式 anchor 接口，并逐窗口核对 pair、目标记录、受试者、起点和原始缓存目标是否与已有评估一致。不存在把隐藏胸导联当模型输入的操作。
- 先按时间拼接完整评分 pair，再用501点（约1秒）移动平均。原始曲线保留 μV，显示版才分别减去完整慢曲线均值；不对目标进行推理校正。
- 只需要完整 task1/task2 缓存和仓库 metadata；无需 demographics、PTB-XL、训练尺度或 GPU。缓存路径来自配置，与训练相同，小写 `data` 的服务器配置保持原状。

## 服务器命令：复用此前三条 V3 记录

本地修改提交并推送后，在服务器拉取 `baseline/B5`。保留服务器自己的路径配置，不执行强制 reset。

```bash
cd /home/qht/huawei
git pull --ff-only origin baseline/B5

# 如现有环境未安装绘图库，只需补这个依赖。
python -m pip install "matplotlib>=3.8"

python -m baselines.B5.plot_anchor_slow \
  --config configs/experiments/b5_local_meta.yaml \
  --evaluation results/B5/E2_finetune_meta/huawei_evaluation \
  --task task2 \
  --leads V3 \
  --pair-id TASK2_695b84d99ccde5 \
  --pair-id TASK2_74acdc093c9e99 \
  --pair-id TASK2_67989297de1432 \
  --slow-seconds 1 \
  --output-dir results/B5/plots/finetune_v3_anchor_slow_v1
```

`--evaluation` 传包含 task1/task2 子目录的目录，不能传 CSV-only 诊断目录。如果评估实际保存在 `best_evaluation`，改成对应真实目录。输出必须是新目录或空目录；重复运行改成 `_v2` 等名字。

删去全部 `--pair-id` 时，默认按去均值慢曲线误差选低、中、高三个代表位置，不自动只挑最差记录。可用 `--max-records 5` 改数量。三条明确的 pair ID 则严格保持上面的顺序。

## 输出与如何看图

每条记录每个导联输出：

- `record_00_V3_anchor_slow_raw.png`：实际输入 I、真实 V3、预测 V3，分三行，共用时间轴，各自独立的 μV 纵轴。
- `record_00_V3_anchor_slow_centered.png`：同样三行，仅为显示各自减去完整记录慢曲线均值；没有幅度缩放。
- `record_00_V3_anchor_slow_curves.npz`：三个未去均值的完整500Hz慢曲线及时间轴，供后续分析。PNG仅以50Hz抽点显示慢曲线，统计仍使用全部500Hz点。
- `record_metrics.csv`：I与真实目标、预测与真实目标的慢曲线相关性、曲线均值/标准差、诊断RMSE。常数信号相关性留空并在报告中写 undefined。
- `plot_manifest.json`、`report.md`：数据来源、核对结果、选取方式与诊断说明。

先看120秒上升/下降是否也在输入 I 中出现，再看几秒到几十秒的局部起伏是否对应。I和V3不必同幅度或同符号，不能要求照抄 I；相关性也不能证明条件可预测性。低相关不证明无法预测，高相关不保证训练集到验证集的泛化。501点移动平均是诊断分解，不是生理基线识别。

```bash
cat results/B5/plots/finetune_v3_anchor_slow_v1/report.md
tar -czf B5_anchor_slow_plots.tar.gz \
  -C results/B5/plots finetune_v3_anchor_slow_v1
```

下载压缩包即可分析，无需上传 checkpoint 或完整预测数组。

## 验证

```bash
python -m unittest baselines.B5.tests.test_anchor_slow -v
```

测试覆盖乱序缓存的身份对齐、错误目标/受试者拒绝、完整记录滤波、显示去均值与原始数据分离、实际 anchor 与预测 I 分离、图像和完整曲线导出、源文件只读、未知 pair/缺失窗口/非空输出拒绝。没有模型训练或参数更新。
