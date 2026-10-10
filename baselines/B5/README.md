# B5：精简微调与最终评估主线

分支 `B5` 从 `baseline/B5` 的 `20f6ea8` 建立。只保留同步 I＋年龄/性别/身高/体重条件的 B5-U 微调、数据预检、正式评估和测试推理。网络仍是原64/128/256条件1D U-Net，FM线性路径和原损失不变；模型/条件/预处理版本不变，兼容已有公开预训练和华为微调 checkpoint。

最终候选是 **EMA权重＋Heun16＋K=16＋seed=42**。K=16是16次独立ODE输出逐点平均，默认配置为 `configs/b5_inference.yaml`。validate/predict自动读取，无需传steps或samples；旧checkpoint内的K=1不会覆盖它。每窗口512次速度场调用，16份依次生成并累加。

旧的从头训练、I-only消融、公开预训练脚本、过拟合、慢走势实验和绘图/诊断代码保存在原 `baseline/B5`。需要重跑公开预训练时使用原分支；本分支加载现成的公开预训练checkpoint微调。共享 `ecg12gen`、metadata和common/preprocessing配置是数据/评分依赖，保留原状。

## 1. 直接评估已有最佳微调模型，不需重新训练

先由用户提交并推送本地新分支，服务器检查没有运行中的旧训练进程或未处理本地改动，再切换：

```bash
cd /home/qht/huawei
git fetch origin
git switch B5
git pull --ff-only origin B5

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_finetune_meta.yaml \
  --checkpoint results/B5/E2_finetune_meta/best.pt \
  --device cuda \
  --output-dir results/B5/E2_finetune_meta/final_B5_heun16_k16
```

首次本地没有B5分支时，fetch后 `git switch --track origin/B5`。不要强制reset；服务器路径修改按实际位置保留。输出目录须新建或为空。终端和输出报告包含r、Task2胸导联RMSE、实际采样设置；`inference_sampling.json`记录配置和checkpoint路径。

使用你已训练好的 `results/B5/E2_finetune_meta/best.pt` 就能复用当前最优候选权重。Git分支不会包含服务器checkpoint和数据；精简代码不意味着从头重训可以自动得到相同权重，也未在本机重算服务器成绩。

## 2. 测试推理

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.predict \
  --checkpoint results/B5/E2_finetune_meta/best.pt \
  --anchor /path/to/visible_synchronous_I.npy \
  --record-id test_record_001 \
  --input-fs 500 --input-unit uV \
  --device cuda \
  --output results/B5/test_predictions/test_record_001.npy
```

anchor只允许可见同步I的 `[T]` 或 `[1,T]` 数组。可选 `--metadata-json /path/to/observed_metadata.json`，内容如 `{"age":24,"gender":"男","height":175,"weight":70}`；没有则按原缺失mask处理。填写真实采样率和单位；默认输出与原输入相同长度、采样率，12导联，μV。输出文件和JSON需不存在。JSON记录实际Heun16/K16。

不会读取真实缺失导联；I输出沿用原辅助预测头，没有默认复制可见I。原始目标与anchor不减median；输出只乘一次冻结scale，不用真实目标基线校正。

## 3. 路径与数据检查

只剩两份实验配置：`b5_base.yaml`为共有参数，`b5_finetune_meta.yaml`为微调入口；不需要选择多个实验版本。`paths.data_root`默认为小写 `data`：

```text
/home/qht/
├── task1_output_v2/
├── task2_output/
└── huawei/
    ├── data/userinfobean.csv
    ├── metadata/
    └── results/B5/
        ├── shared/preprocessing_scales.npz
        ├── E2_public_meta/best.pt      # 仅需要继续微调时
        └── E2_finetune_meta/best.pt    # 直接评估/测试使用
```

不需上传或读取PTB-XL原始数据来评估/推理。沿用原缓存、受试者划分、单位和尺度；文件夹大小写按实际配置。

```bash
python -m baselines.B5.prepare \
  --config configs/experiments/b5_finetune_meta.yaml \
  --report results/B5/preflight_finetune.json
```

预检不训练、不下载；ready=true才表示缓存/尺度/人口学检查通过。已有冻结尺度不要重新拟合。validate/predict从checkpoint读取其冻结尺度，prepare与微调读取配置的既有尺度文件并检查兼容。

## 4. 可选继续微调

训练不是获得当前候选成绩的必需步骤。若另开一轮微调，先把配置中 `paths.output_dir` 改为新的空目录，然后：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_finetune_meta.yaml \
  --init-checkpoint results/B5/E2_public_meta/best.pt \
  --device cuda --execute-training
```

也可用兼容微调best初始化新的微调运行。精确续训用 `--resume /path/to/last.pt`，不能同时传init-checkpoint，保留原阶段/损失/数据/关键训练设置；只有显式execute-training才训练。

周期验证保持原实验Heun16/K1，best.pt仍按Task1 r选择，last.pt用于续训。独立最终评估/测试用K16，选模历史不伪装成K16。另开新实验若要按K16选模型，可在新实验配置改sampling.samples，但不能把旧K1最佳分数作为新设置的历史基准。

## 5. 环境与验证

Python>=3.10，沿用BioFlow；依赖是NumPy、PyYAML、PyTorch、SciPy，不再需要WFDB或matplotlib来使用这条主线。

```bash
python -m unittest discover -s baselines/B5/tests -v
```

测试不运行优化器更新，覆盖网络/条件/缺失mask、原始电压尺度、ODE、旧checkpoint、默认K16、参数覆盖、推理尾段、采样率和main评分合同。best-score保存方式、共享main、网络参数布局均未修改。
