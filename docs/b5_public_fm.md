# B5公开数据扩充：FM-C / FM-P操作说明

只扩大FM的公开训练数据。慢CNN使用冻结的旧慢PCC20，不参加反向传播。
本分支从B5-best的`ec12008`创建，沿用可靠I、目标导联QC、目标记录等权、Task2两设备等权的最新平均r协议。
历史0.698216属于此前参考范围，不能直接当成本次新协议基准；下面会生成同协议baseline。

本地只执行合成CPU功能/回归检查及脚本语法检查，未启动真实数据训练。真实数据的完整审计、GPU数值和实际得分须在服务器按下述步骤确认。

## 1. 两组到底做什么

| 项目 | control（FM-C） | expanded（FM-P） |
|---|---|---|
| 初始化 | 同一份已有PTB-XL公开FM的EMA权重 | 同左 |
| 公开继续训练 | 仅PTB-XL train | PTB-XL train＋Chapman train |
| 公开预算 | 20,000次更新，batch8，LR 1e-4 | 同左 |
| 公开阶段选权重 | 固定第20,000次更新EMA，不按公开r挑轮次 | 同左 |
| 华为微调 | 从本组public_final.pt重新建立优化器，100轮 | 同左 |
| 华为学习率/损失/网络 | 原B5设置；初版workers0、全精度、累积1 | 同左 |
| 周期验证 | epoch0及每5轮，Heun16/K16，seed42 | 同左 |
| 选模 | FM快残差＋0.75×固定慢预测，完整120秒平均r | 同左 |

20,000次更新是起始预算，不是已验证最优参数。混合概率默认按审计后的两来源train记录数计算；不会强制1:1。
可在**prepare前**编辑`configs/b5_public_fm.json`；若指定概率，prepare加`--chapman-probability 0.5`。准备后不要修改设置/代码再续跑旧study，应新建run。

## 2. 下载数据与目录

新增数据：[PhysioNet完整45,152患者版本](https://physionet.org/content/ecg-arrhythmia/1.0.0/)。
不要额外合并早期10,646例版本，避免重复。下载解压保留`.hea/.mat`相对目录结构。

```text
/home/qht/huawei/
├── data/
│   ├── PTB-XL/
│   │   ├── ptbxl_database.csv
│   │   └── records500/...
│   └── Chapman_Shaoxing_Ningbo/
│       └── WFDBRecords/.../*.hea 与 *.mat
├── metadata/                 # 当前main对应的划分、QC、人口学等
└── results/                 # 现有公开权重、模型包和本次新结果
```

如果PTB-XL或下载解压后的目录实际不同，prepare时用真实路径，不要重复下载或移动现有数据。
运行需要服务器现有BioFlow环境中的torch、numpy、PyYAML、wfdb（WFDB的MAT读取依赖scipy）。不自动下载、安装依赖，不自动git fetch。

## 3. 服务器首次创建工作树

先把本地实验分支提交并推送到服务器可访问的gitee，再执行一次：

```bash
cd /home/qht/huawei
git fetch gitee B5-best-exp-PublicFM
git worktree add -b B5-best-exp-PublicFM \
  /home/qht/huawei_B5_best_exp_public_fm \
  gitee/B5-best-exp-PublicFM
```

以后更新代码，只有在**尚未prepare新实验**时执行：

```bash
cd /home/qht/huawei_B5_best_exp_public_fm
git fetch gitee B5-best-exp-PublicFM
git merge --ff-only FETCH_HEAD
```

所有后续命令在此新工作树运行。数据仍从`/home/qht/huawei`读取，稳定B5-best不切分支、不改模型。

## 4. 准备共享数据（一次，CPU，不训练）

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_exp_public_fm
sed -i 's/\r$//' scripts/run_b5_public_fm.sh
bash -n scripts/run_b5_public_fm.sh

bash scripts/run_b5_public_fm.sh prepare \
  --ptbxl-root /home/qht/huawei/data/PTB-XL \
  --chapman-root /home/qht/huawei/data/Chapman_Shaoxing_Ningbo
```

默认run为`/home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1`。
默认慢CNN从稳定配置固定的已验证组合包复制：

```text
/home/qht/huawei/results/B5-best-cross/seed42_20261017_094115_417122/control100__pcc20.pt
```

默认公开初始化**不猜文件名、不选最近权重**：读取原华为checkpoint
`results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt`
的`initialization.source_checkpoint`及SHA，定位当时使用的公开权重。
如果该路径已移动或不是这次想用的公开权重，在上面prepare命令中增加：

```text
--public-checkpoint /真实路径/公开训练结果/best.pt
```

必须是`stage=public`、同架构同尺度的FM，不能传Huawei best或组合包。
如慢模型已换位置，可传`--fixed-bundle /真实路径/组合包.pt`，整个实验始终使用复制的同一慢权重。

prepare会：

- 要求发现完整45,152个Chapman头文件，检查所有记录及依赖文件完整；缺文件/不支持格式会报错，不静默跳过。
- WFDB按头文件gain/baseline读取物理电压并统一μV，规范导联顺序；不逐窗去均值，不额外去漂移、不裁振幅。
- 基本QC标记非有限、近乎平直、长时间常值、ADC打满的坏导联；记录须有可靠I且至少6导联有效。
- 坏导联在缓存中填0仅保证预处理有限，loss用mask排除，绝不以填0后的信号作为正常标签评分。
- PTB-XL保留fold1–8/9/10；新增数据用固定患者ID哈希80/10/10划分，test不参与训练或调参。
- 完全相同的物理波形去重，优先保留PTB-XL及其held-out记录，避免训练副本泄漏。
- 官方新增库没有独立的跨记录患者映射表，默认按其“每患者ECG”的record ID划分，并在报告明确记录这一假设；如另有患者映射，传`--chapman-patient-map map.csv`，字段`record_id,patient_id`。
- 不承诺检测出近重复或未公开的跨库同患者关系。QC是基本技术检查，不是临床判读。
- 将公开波形转成独立只读使用的float32缓存，冻结模型、配置、数据指纹；原始数据不改。

这一步会花时间和磁盘：公开缓存约十几GB；100轮两组每5轮保存raw FM、组合预测和标签，整个实验建议预留**至少120GB**，实际取决于验证配对数。
查看`shared/public/audit_summary.json`、`shared/train_distribution.json`、`shared/study.json`中的有效量、排除原因、振幅分布与实际混合概率后再训练。
准备完成不要重复执行prepare。输出已存在会拒绝覆盖；若审计失败，修好下载后用新的`--run /绝对路径/新名字`重新准备。

## 5. CPU预检（一次，不训练）

```bash
bash scripts/run_b5_public_fm.sh check
```

结果：`preflight.json`应有`ready=true`、`same_public_initial_weights=true`、`optimizer_updates=0`。
检查完整5000点前向，公开两来源和华为样本接口一致，target I与anchor采用同一尺度。

## 6. 当前稳定版按本次协议算基准（一次，GPU）

```bash
bash scripts/run_b5_public_fm.sh baseline --gpu 1
```

这是生成评估，不训练。用复制的当前稳定FM＋慢PCC20在**当前可靠I/QC评分范围**算基准，输出`baseline/correlation_score.json`。
不要把旧协议的0.698216直接填进新报告。无需等待新的慢CNN完成。

## 7. 两个终端并行：各组公开训练完自动接华为微调

终端A，GPU0：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_exp_public_fm
bash scripts/run_b5_public_fm.sh train --group control --gpu 0
```

终端B，GPU2：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_exp_public_fm
bash scripts/run_b5_public_fm.sh train --group expanded --gpu 2
```

用不同终端/tmux会话保持运行。两组各自顺序执行public→公开验证→Huawei；互不等待，最终自动导出best/last组合包。
脚本GPU是物理卡号，程序内统一`cuda`，不会错误写成`cuda:2`。
日志在run根目录的`control.log`、`expanded.log`，不写进要求空目录的阶段输出。

公开结束后会在每个来源固定前64条validation记录上做Heun16/K16真实生成诊断；这不是正式华为评分，也**不用于公开权重选择**。
两组数据抽样RNG独立于FM噪声；报告记录访问次数、不同记录覆盖率，不能只用epoch推算数据见过多少。

## 8. 中断续跑

确定原进程已停止后，原命令加`--resume`：

```bash
bash scripts/run_b5_public_fm.sh train --group control --gpu 0 --resume
```

```bash
bash scripts/run_b5_public_fm.sh train --group expanded --gpu 2 --resume
```

恢复各阶段`last.pt`中的raw权重、EMA、优化器、随机状态及抽样进度，不重新初始化。
公开阶段每2000更新保存一次；华为每轮保存last，可能重做最后未保存部分。完成的公开阶段验证一致后复用，不再训练。
学习率、数据、代码或预算变化需要新study，不用旧last继续。不能并发启动同一个组；同一group只有一个训练进程。

## 9. 完成后比较与打包

```bash
bash scripts/run_b5_public_fm.sh compare
bash scripts/run_b5_public_fm.sh pack
```

汇总：`/home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/comparison/summary.csv`

发回的压缩包：**`/home/qht/huawei/B5_PublicFM_reports.tar.gz`**。
只打包配置、审计摘要/CSV、日志和评分，不包含pt/npy/原始ECG。

## 10. 权重、部署与以后换慢CNN

```text
public_expansion_v1/
├── shared/                 # study、共同公开初始化、固定慢组合包、公开数据缓存
├── baseline/               # 同协议稳定版基准
├── control/
│   ├── public/public_final.pt
│   └── huawei/
│       ├── best.pt / best_r.pt / last.pt      # FM训练checkpoint
│       ├── best_bundle.pt / last_bundle.pt    # 两网络自包含推理包
│       ├── candidates/epoch_0000.pt ... epoch_0100.pt
│       └── evaluations/epoch_0000/ ... epoch_0100/
└── expanded/               # 同结构
```

`best.pt`是在当前固定慢CNN下组合r最高的FM候选，包括epoch0；不是最优公开checkpoint，也不是纯FM输出r最高。
`best_bundle.pt`可直接交给B5-best的validate/predict，无需原始训练权重/真实目标作为推理条件。
如要复核扩充组导出包的实际部署入口：

```bash
python -m baselines.B5_best.validate \
  --config /home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/shared/data_config.json \
  --checkpoint /home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/expanded/huawei/best_bundle.pt \
  --device cuda \
  --output-dir /home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/expanded/deployment_check
```

上面的部署核对需要先在该终端设置`export CUDA_VISIBLE_DEVICES=2`；与训练并发时自行选择空闲卡。
训练保存每个候选的原始FM预测`fm_prediction_uV.npy`，更换慢CNN不必重训FM：

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.B5_public_fm.recombine \
  --study /home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/shared/study.json \
  --evaluation /home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/expanded/huawei/evaluations/epoch_0100 \
  --slow-bundle /真实路径/包含新慢CNN的组合包.pt \
  --output /home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/expanded/new_slow_epoch100 \
  --device cuda
```

这只是重评一个指定候选。要找新慢CNN下最好的FM轮次，分别对两组所有`epoch_*`候选重评，不能只换原best就称为新搭配全程最优。
新慢CNN须已导出为兼容的S50组合包、使用相同尺度，仍固定beta0.75。只计算新慢预测和组合分数，不重新生成FM。
