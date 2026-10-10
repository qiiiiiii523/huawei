# B5 最终评估候选：Heun 16步，K=16

默认配置在 `configs/b5_inference.yaml`：

```yaml
sampling:
  seed: 42
  solver: heun
  steps: 16
  samples: 16
```

`validate`（正式独立评估）和 `predict`（仅可见输入的测试推理）都自动读取它，无需传 `--steps 16 --samples 16`。这包括以前保存了 K=1 的 checkpoint；实际采样设置取自当前推理配置，不会被 checkpoint 的旧设置覆盖。

K=16 表示同一条件下生成16份缺失导联波形，然后逐点求平均；每份使用Heun16，合计每窗口512次速度场网络调用。固定seed=42、窗口key和sample index，前4/8次样本与对应K=4/8设置一致。16份依次生成并累加，不一次堆积16份状态；计算耗时增加，瞬时状态内存不按K倍增。

## 配置优先级

显式 `--steps/--samples/--solver/--seed` 参数 > `--sampling-config` 指定文件（未指定时用上述默认文件）。因此日常不必传采样参数，研究对照仍能明确设置K=1。

`validate --config` 继续定义网络、数据路径和验证集；其 `sampling` 在独立评估时由推理配置替换。`predict` 从 checkpoint 读取模型和冻结尺度，从推理配置读取生成设置。

训练过程的周期验证仍使用 `configs/experiments/b5_*.yaml` 中原有sampling，不受本推理配置影响。原best.pt选择指标仍为Task1 r，不重新训练或补回历史权重。旧best按K=1选出，不保证也是K=16最佳轮次。若另开新实验希望按K=16选模型，显式在该新实验配置设sampling.samples=16，注意验证时间增加；不要以旧的K=1 best分数续接新K设置后声称历史一致。

不改 raw μV 目标、median、scale、网络、损失、main 或 B6。不要直接用推理配置运行train_huawei，它不是完整训练配置。

## 服务器：评估已有微调最佳模型

本机提交推送后，在服务器保留自己的数据路径修改并拉取：

```bash
cd /home/qht/huawei
git pull --ff-only origin baseline/B5

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_finetune_meta.yaml \
  --checkpoint results/B5/E2_finetune_meta/best.pt \
  --device cuda \
  --output-dir results/B5/E2_finetune_meta/final_heun16_k16
```

输出目录必须新建或为空，重复评估换新名字，不覆盖历史K=1报告。终端显示实际sampler、Task1/Task2 r和Task2胸导联RMSE；目录内新增 `inference_sampling.json` 记录实际配置，各任务报告同时记录steps、samples、solver和NFE。

## 服务器：测试时直接预测

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.predict \
  --checkpoint results/B5/E2_finetune_meta/best.pt \
  --anchor /path/to/visible_synchronous_I.npy \
  --record-id test_record_001 \
  --input-fs 500 --input-unit uV \
  --device cuda \
  --output results/B5/test_predictions/test_record_001.npy
```

这里anchor仅为可见同步I，形状 `[T]` 或 `[1,T]`，不是12导联真实目标。可选加 `--metadata-json /path/to/observed_metadata.json`；没有时按现有缺失mask处理。输出JSON包含实际sampling和配置路径，默认就是Heun16/K16。输出文件和JSON均须不存在。

若anchor采样率不是500Hz或单位不是μV，填写真实参数；默认输出与原输入相同采样率。无需传sampling-config、steps或samples，也无需改旧checkpoint。

## 复现旧评估

在相同validate/predict命令末尾加 `--samples 1`。对照32步只加 `--steps 32`。这两项现在都支持显式覆盖，默认参数不变。

## 功能检查

```bash
python -m unittest baselines.B5.tests.test_inference_sampling baselines.B5.tests.test_b5 -v
```

只做CPU单元检查，不启动训练。覆盖旧K=1 checkpoint的默认覆盖、显式参数优先级、非法配置拒绝、验证/预测入口的一致设置、输出JSON记录、原实验sampling不变。
