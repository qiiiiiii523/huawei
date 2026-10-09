# B5-U：同步 I＋人口学条件 Flow Matching

第一版主线实现。代码不会在 import、数据检查或缺少执行开关时启动训练，也不会自动下载数据。

## 已实现的范围

- 三尺度条件 1D U-Net，64/128/256 通道，11 导联速度头和独立辅助 I 重建头。
- 同步 I 多尺度特征注入，连续时间编码，年龄/性别/身高/体重 MLP＋FiLM，字段 mask、整组人口学 dropout 和单字段 dropout。
- 独立噪声—目标线性 Flow Matching、坏目标导联屏蔽、Huber/PCC/I 辅助监督。
- Heun/Euler 真正 ODE 采样、多样本均值、按记录/窗口稳定噪声种子。
- 华为 train-only 同步缓存、当前 main 的验证 Dataset 与记录级 evaluator。
- 本地 PTB-XL records500 读取、WFDB 物理单位和导联映射、患者 fold 检查、下载完整性检查。
- 公开从零预训练、华为从零基准、公开 checkpoint→华为微调，以及兼容性校验和精确续训。
- 保存 EMA、优化器、随机状态、数据清单摘要、条件版本、尺度参数和采样配置。
- 仅可见同步 I＋可选人口学的任意长度预测、尾窗 padding/裁去、输入/输出采样率适配。

首版不实现历史 ECG/PPG 条件融合，也不启用 CNN＋Transformer。现有网络和条件 schema 保留为稳定基础版；后续扩展需要新 schema、真实配对数据和加载审计，不能把缺失 mask 当作该模态已经学习。

## 与当前 main 对接

预处理直接复用 ecg12gen.preprocessing：

- anchor I 与 d12 target 只除冻结 scale，不减 median。
- scale 仅由华为 train-only 去重同步索引拟合，公开预训练与华为微调复用同一文件。
- B5-U 不使用跨时间 context，因此首版不需 context 的尺度；主 Dataset/评估协议不改。
- 训练完整 d12 辅助输出中 I 由重建头预测，不拿真实 I 回填训练输出。
- 输出只乘回一次 scale，不加 target baseline；evaluator 不中心化、不替换 I。
- r_missing11 评分 II–V6；**当前 main 的 Task2 RMSE 加分只统计 V1–V6**，missing11_mean_rmse_uV 仍是 11 导联平均误差。直接调用 main evaluator，不另写计分公式。

一个基础模型会同时报告 task1、task2。默认配置以 **task1 validation r_missing11** 选择 best.pt，task2 另行报告；这避免隐式更改 main 的 checkpoint 指标。若研究 Task2 专用 checkpoint，请改 validation.selection_task=task2，并用独立 output_dir，所有对照同样设置。公共预训练以 PTB-XL fold9 的 r_missing11 选 checkpoint，该结果不作为华为竞赛成绩。

## 目录和公开数据

默认配置从仓库目录计算：

```text
HW/
  huawei/
    baselines/B5/
    configs/experiments/b5_*.yaml
  task1_output_v2/
  task2_output/
  Data/
    userinfobean.csv
    ptbxl_database.csv
    records500/00000/00001_hr.hea
    records500/00000/00001_hr.dat
    ...
```

PTB-XL 也可位于 Data/ptb-xl/1.0.3 等子目录，适配器会在最多三层路径中寻找数据库 CSV；多个版本时必须显式设置 paths.ptbxl_root。已下载 records100 并不表示可以开始本方案的 records500 预训练，不自动将 100 Hz 波形插值当作 500 Hz 高分辨率数据。

缺文件时 preflight 返回 ready=false 和退出码 2，训练构造器拒绝静默使用不完整子集。不会调用 wget、请求远端 WFDB 或自动下载。

移动到训练服务器后调整配置中的 paths.data_root、paths.huawei_data_root、paths.ptbxl_root。保留患者级 split、500 Hz 原始电压缓存与 main metadata；不要重新按窗口随机划分。

## 环境

Python >=3.10。先在实际训练机器按 CPU/CUDA 环境安装 PyTorch，然后：

```powershell
python -m pip install -r baselines/B5/requirements.txt
```

根目录 requirements.txt 只描述公共 main 的 NumPy/PyYAML，B5 的 Torch/WFDB/SciPy 依赖位于自己的 requirements.txt。无需在当前无 GPU 的本机安装训练环境。

## 第一步：只做准备检查，不训练

以下命令从 huawei 仓库根目录执行。

```powershell
python -m baselines.B5.prepare --config configs/experiments/b5_local_meta.yaml --fit-scales --report results/B5/preflight_huawei.json
python -m baselines.B5.prepare --config configs/experiments/b5_public_meta.yaml --check-public --report results/B5/preflight_public.json
```

第一条只从华为训练数据拟合尺度并检查缓存，不更新任何网络参数。尺度文件已存在时去掉 --fit-scales，直接复用；禁止微调时重新拟合或对验证目标拟合。

第二条检查下载和单位。公共十秒记录本身合法，公共适配器不会套用华为原始记录“少于30秒不训练”的规则。PTB-XL folds1–8 train、9 validation、10 test；同一 patient_id 跨 fold 会报错。

元信息冲突逐字段设缺失，不随意选最后一行；找不到患者时所有人口学字段缺失。ID 只用于关联/划分，不输入模型。PTB-XL sex=0为女、1为男，统一映射到 B5 的 male=0、female=1、unknown=2；匿名高龄编码归入 90+，不会直接把 300 岁当数值条件。

## 第二步：训练服务器先做小集合过拟合诊断

先完成华为缓存/环境检查和 train-only 尺度准备，不必等公开数据下载完。下面命令只应由用户在训练机器上显式执行，本次交付未运行任何优化步骤：

```bash
python -m baselines.B5.overfit --config configs/experiments/b5_local_meta.yaml --windows 1 --steps 1000 --device cuda --output-dir results/B5/overfit_w1 --execute-training
python -m baselines.B5.overfit --config configs/experiments/b5_local_meta.yaml --windows 4 --steps 1000 --device cuda --output-dir results/B5/overfit_w4 --execute-training
```

诊断仅从 Huawei train-only 索引选择不同记录、完整可靠且非恒定的目标窗口，固定可见条件和 target，每次优化仍随机采样 t/noise。关闭人口学 dropout/weight decay，评估使用固定噪声的真实 ODE 生成。输出 diagnostic.jsonl、training_target_uV.npy、last_prediction_uV.npy；不保存可复用的 checkpoint。

关注平均训练 loss 趋势，以及 training_window_r_missing11、training_window_mean_rmse_scaled 相对 step0 是否显著改善。单窗口可将 r 接近0.95以上、缩放空间RMSE接近0.1以下作为参考，但不设成所有数据必达的硬门槛。随机 FM 的瞬时 loss 不必为零；波形仍不对、训练样本都无法拟合时，先查单位、对齐、mask和采样，不立即跑公开全量。

这是训练集合上的窗口诊断，不是验证成绩或正式记录级比赛得分。overfit 同样必须带 --execute-training；当前开发与CPU检查没有运行它的优化循环。

## 第三步：将来在训练机器上执行完整训练

**本次交付没有执行本节任何训练命令。只有显式传 --execute-training 才会启动训练。** 不带开关会在导入训练运行时之前报错。

### E1：华为从零基准

```powershell
python -m baselines.B5.train_huawei --config configs/experiments/b5_local_meta.yaml --device cuda --execute-training
```

### E2：公开预训练→华为微调主线

```powershell
python -m baselines.B5.train_public --config configs/experiments/b5_public_meta.yaml --device cuda --execute-training
python -m baselines.B5.train_huawei --config configs/experiments/b5_finetune_meta.yaml --init-checkpoint results/B5/E2_public_meta/best.pt --device cuda --execute-training
```

公开预训练从随机初始化开始，同步 I、人口学和主干一起学习；华为微调加载公开模型的 EMA 权重并继续训练全部基础参数。同一数值/条件定义跨阶段保持一致。微调必须给 compatible checkpoint；不会静默退回随机初始化。

### E3：人口学与框架对照

```powershell
python -m baselines.B5.train_public --config configs/experiments/b5_public_i.yaml --device cuda --execute-training
python -m baselines.B5.train_huawei --config configs/experiments/b5_finetune_i.yaml --init-checkpoint results/B5/E3_public_I/best.pt --device cuda --execute-training
python -m baselines.B5.train_huawei --config configs/experiments/b5_local_i.yaml --device cuda --execute-training
```

E1与E2比较公开预训练；E2与公开I-only路线比较人口学条件；local-I与相同骨干/数据/预算的I-only diffusion比较FM。不能将额外公开数据和人口学增益全部归因于FM。

### 精确续训

```powershell
python -m baselines.B5.train_huawei --config configs/experiments/b5_finetune_meta.yaml --resume results/B5/E2_finetune_meta/last.pt --device cuda --execute-training
```

--resume 恢复模型、EMA、优化器、GradScaler、随机状态和数据加载顺序；必须保持数据清单、阶段、选模人群和关键训练设置。增加总 epochs 可继续；换数据/条件/预算应另建配置与 output_dir 并使用 --init-checkpoint。精确续训需同 CUDA 设备数量。

所有配置中的 epoch/LR 是起始设置，不是已调优结果。首版无 LR scheduler、回放混训或历史分支，这些不是隐藏启用的功能。梯度累积处理最后一个不足整组的 batch，验证始终用 EMA 和实际 ODE 生成。

## 第四步：验证已有 checkpoint

```powershell
python -m baselines.B5.validate --config configs/experiments/b5_finetune_meta.yaml --checkpoint results/B5/E2_finetune_meta/best.pt --device cuda --output-dir results/B5/E2_final_validation
```

输出每任务的 prediction_uV.npy、target_uV.npy、window_metadata.csv、overall_metrics.csv、lead_metrics.csv、report.md，以及两任务计分汇总。public validation 会标记 source_domain=PTB-XL，不能当华为测试成绩。输出目录需为空，避免覆盖先前记录。

验证不将真实 target 用于生成。score 由 main evaluator 计算，记录级拼接检查会对缺窗/重复窗报错。Task2 缓存无 expected_window_count 时的末尾缺窗判定限制仍按 main 实现，不宣称已自动补齐缓存。

## 第五步：只用可见同步 I 做预测

```powershell
python -m baselines.B5.predict --checkpoint results/B5/E2_finetune_meta/best.pt --anchor visible_record_I_uV.npy --metadata-json observed_metadata.json --record-id record_001 --input-fs 500 --input-unit uV --device cuda --output results/B5/record_001_prediction.npy
```

observed_metadata.json 示例：

```json
{"age":24,"gender":"男","height":175,"weight":70}
```

没有人口学信息时省略 --metadata-json。anchor 只接受 [T] 或 [1,T] 的真实可见同步 I；不会读取完整 d12/隐藏目标。长度不限十秒，尾段 edge padding 后预测并裁去 padding。默认输出恢复到输入采样率及原始长度，单位 μV，导联顺序标准 d12；可用 --output-fs 明确指定其他接口输出采样率。

默认 I 通道来自辅助重建头，与训练/验证一致。只有正式接口明确需要复制可见 I 时才用 --copy-observed-i，此选项仅在提交预测入口，不参与 evaluator 替换或选模。

可覆盖 --steps、--samples、--solver、--seed 做独立采样实验。Heun16/K1约32NFE；K4约128NFE。噪声基于record-id＋窗口起点，不依赖batch排序。禁止用真实target挑最好样本。

输出 .npy 与 JSON sidecar 使用新文件名，不覆盖输入或先前结果。预测格式是通用 NPY 适配器；最终主办方可执行包装/字段格式需要在正式接口确认后另行接入，不能将其称为已完成官方打包提交。

## 无训练的 CPU 检查

```powershell
python -m unittest baselines.B5.tests.test_b5 -v
```

检查完整5000点网络前向、FiLM/mask、坏导联屏蔽、ODE解析例、按记录噪声、真实WFDB格式夹具、患者fold隔离、尺度/安全checkpoint加载、尾段与采样率恢复。损失只做一次导数检查，不创建优化器或更新参数，不进入训练循环。

可选实际缓存的只读集成检查：

```powershell
$env:B5_REAL_HW_ROOT = 'C:\Users\Ashley\Desktop\HW'
python -m unittest baselines.B5.tests.test_b5 -v
```

只从真实缓存读取/在内存计算train尺度，并用target自身检查main评分契约；这不是模型性能结果，不拟合验证统计，不写回Data。

checkpoint使用 weights_only=True 加载，且检查architecture/schema/scale校验值。生成的checkpoint、尺度、波形预测、日志均放results下，现有.gitignore已排除；不要提交原始数据或训练产物。
