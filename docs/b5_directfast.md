# B5-final-exp-directfast：直接回归快分量实验

## 1. 分支和实验范围

从 `B5-final@9f3e9623cff398b6687969f19b556df50123cdfc` 创建独立分支 `B5-final-exp-directfast`。
本地工作树：`C:/Users/Ashley/Downloads/备赛/B5-CMF工作树/Final-DirectFast`。
与 `B5-final-exp-slow120` 平行，原Final入口、配置及权重均保留。

本分支新增独立模块 `baselines.B5_directfast`，不改变原Final默认模型。
训练和运行均在服务器执行。本地只做静态源码检查，未启动测试、前向、推理或训练；当前没有新模型分数。

| 组 | 内容 |
|---|---|
| F0 | 原Final固定预测，完整组合平均r应复现0.7075136581 |
| F1 | 原FM的EMA初始化，按原FM目标继续训练；P固定 |
| D1 | 同一EMA初始化、同样U-Net结构，固定状态/时间为0，重新训练直接回归；P固定 |

F1/D1匹配中心训练窗口、更新次数、学习率、训练seed和选模频率，不宣称计算成本一致。
D1还会分别编码相邻窗口，为去慢提供真实相邻预测；每次网络仍只看10秒。
两组新训练均使用在线权重选模，不另加EMA；包含第0步候选。
D1第0步不等于Final。F1第0步应该接近原FM；脚本会在服务器核对。

## 2. 核心实现

```text
同步I＋人口学 → DirectFastNet → 每10秒Q
12个Q按时间拼接 → H(Q)=Q−501点滑动平均(Q)
H(Q)＋0.75×原P慢预测 → 原机器d6 / 显式冻结body迁移
```

网络复用现有64/128/256三尺度U-Net及人口学编码，从Final基础包的FM EMA初始化。
新任务直接训练Q，经H后监督快分量，不计算FM速度场损失，不进行ODE积分。
原P始终冻结；输出I使用原冻结FM条件编码/I头，不参与D1训练。
F1训练保留原FM辅助I损失，但最终输出I也使用冻结原I头，避免改变输出合同。

训练D1时，分别预测中心及相邻10秒窗口，拼接后对中心有效区计算可微501点去慢。
记录端点用edge padding；窗口交界使用真实相邻预测。
目标快分量从真实完整记录计算一次。训练归一化沿用原尺度。
部署拼完整120秒后执行一次H；不再调用旧combine对快结果重复去慢。

默认D1损失仅为mask后的快Huber。`direct_loss.combo_pcc`默认0；若要检验与原P组合后的PCC监督，复制实验配置后独立开一轮，不混入首轮结论。

Final推理包刻意不包含FM训练损失。F1默认读取原Final历史run_config中的loss，并在study中保存配置SHA和来源。
若保留了原FM完整checkpoint，可传`--fm-source-checkpoint`；其文件SHA必须与Final包中provenance.fm_sha256一致，随后读取该文件loss_config。
两种来源都明确记录；不能把所用配置不一致的实验称为单因素框架对照。

## 3. 在服务器建立工作树

先把本地该分支推送到自己的远端，或者使用本次交付的代码zip上传到独立目录。
以下命令仅用于服务器首次创建（目录和分支未被使用时）：

```bash
cd /home/qht/huawei
git fetch gitee B5-final-exp-directfast
git worktree add --detach /home/qht/huawei_B5_final_exp_directfast gitee/B5-final-exp-directfast
cd /home/qht/huawei_B5_final_exp_directfast
conda activate BioFlow
```

已存在该工作树时，先检查改动，再fetch和更新；不要reset/clean覆盖实验内容。
zip需完整解压到独立代码目录；不要覆盖原`/home/qht/huawei`。
同一Python环境继续复用现有项目依赖，无需本地安装新训练环境。

仍需保留服务器已有的原权重、数据和基础预测缓存：

```text
/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt
/home/qht/huawei/results/B5-best-context/newFM_body_adapt_v1/adapt/best.pt
/home/qht/huawei/results/B5-best-context/FM_P_best45_B43_v1/base_inference
```

这些文件只读使用，代码zip不包含.pt或真实数据。
仓库/结果根目录搬迁时用`--repository-root`；权重单独搬迁可指定原Final的`--checkpoint`、`--body-checkpoint`，SHA仍固定。
历史run_config若搬迁，使用`--config /绝对路径/run_config.yaml`，其内部数据路径也应有效。

## 4. 建议先做服务器预检

以下命令会在服务器执行前向、梯度检查和F0评分，但不会更新优化器：

```bash
bash scripts/run_b5_directfast.sh \
  --repository-root /home/qht/huawei \
  --gpu 1 \
  --run /home/qht/huawei/results/B5-final-exp-directfast/preflight_01 \
  --prepare-only
```

使用新的、未存在的run目录。脚本读取原基础缓存，先复现原Final四路结果，随后生成新缓存，进行服务器预检，再评估本分支F0。
预检检查D1确定性、H与NumPy边界一致、有效梯度、冻结I头、原P复现及训练快目标。
验证I从live observed-input字段读取；不从隐藏目标取I作为验证预测输入。
患者划分、完整12窗、QC及原始数据指纹不一致时停止。

若基础缓存缺失，可加`--regenerate-base`在本轮新目录重新生成，耗时明显增加。
已有基础缓存可用`--baseline-evaluation /绝对路径`指定；两者不能同时指定。

## 5. 一条命令完整运行

```bash
bash scripts/run_b5_directfast.sh \
  --repository-root /home/qht/huawei \
  --gpu 1 \
  --run /home/qht/huawei/results/B5-final-exp-directfast/run_01
```

这会重新准备一份独立缓存。若已经做完上一节预检并想复用它，按第6节在原预检run内分别执行训练和后续命令。

默认配置 `configs/b5_directfast.json`：训练10轮、seed43、LR=1e-4、梯度累积4个中心窗口、每2轮及最后一轮验证。
这些只是首轮预算，不是已验证最优超参数。F1验证需真实Heun16/K16，明显比D1慢。
显存不足时可减小`--batch-size`（默认4，主要影响验证/推理）；D1训练每次最多3个窗口，进一步调整需另行修改设计并记录。

需要改变epochs或seed时，先复制配置到新文件并修改，再用`--experiment-config`传入。
准备时会把完整配置固化在study.json；不能在已有训练中悄悄换配置。

流程依次是：资产核对→原Final复现→完整记录缓存→服务器预检→F0→F1训练→D1训练→两组best/last复评→两组best打包→推理包从原始I复评→汇总。
默认按`baseline`完整120秒r选best，固定报告`baseline/d6_only/both`三条路由。

## 6. 复用预检结果或中断后继续

先设定原run绝对路径，并固定与第一次相同的GPU可见设置：

```bash
cd /home/qht/huawei_B5_final_exp_directfast
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run=/home/qht/huawei/results/B5-final-exp-directfast/preflight_01
```

若F1/D1目录尚未创建，逐组运行：

```bash
python -u -m baselines.B5_directfast.train --run "$run/shared" --group F1 \
  --output "$run/F1" --device cuda --execute-training
python -u -m baselines.B5_directfast.train --run "$run/shared" --group D1 \
  --output "$run/D1" --device cuda --execute-training
```

某组中断且已有last.pt时，只续该组；例如：

```bash
python -u -m baselines.B5_directfast.train --run "$run/shared" --group D1 \
  --output "$run/D1" --resume "$run/D1/last.pt" --device cuda --execute-training
```

只支持同组、同输出目录、同缓存和源码的epoch边界续跑，恢复优化器、数据顺序和CPU/CUDA RNG。
中途尚未保存的epoch会重跑；已完成的组不再续跑。
若中断发生在best更新但last未更新之间，程序会因best较新而停止，保留现场，需先核对checkpoint再处理，不能直接覆盖。

训练完成后，执行下面的复评和打包（各输出目录应尚未存在）：

```bash
for group in F1 D1; do
  for name in best last; do
    python -u -m baselines.B5_directfast.evaluate --run "$run/shared" \
      --candidate "$run/$group/$name.pt" --output "$run/verification/$group/$name" --device cuda
  done
  python -u -m baselines.B5_directfast.package --run "$run/shared" \
    --candidate "$run/$group/best.pt" --output "$run/packages/${group}_best.pt"
  python -u -m baselines.B5_directfast.evaluate --run "$run/shared" \
    --package "$run/packages/${group}_best.pt" --output "$run/verification/$group/package" --device cuda
done
python -m baselines.B5_directfast.summarize --run "$run" --output "$run/comparison"
```

已成功生成的复评/打包不要重复覆盖；跳过已完成步骤，或者给新复评指定新的输出目录并自行核对。
汇总入口读取上述标准目录。

## 7. 如何看结果

优先查看 `comparison/summary.md`：

- `baseline`：快支路本身有无收益，F0基准约0.705557。
- `d6_only`：增加原机器修正后的表现，F0约0.707297。
- `both`：冻结body显式迁移后的完整结果，F0约0.707514。

新快支路会改变body输入分布；body来源SHA仍指原Final，日志明确标记`frozen_body_transfer`并记录新模型状态SHA，没有关闭或篡改原Final校验。
如果baseline提高而both下降，先分析冻结body迁移；不把两者混为快网络失败。
只有D1在同一路由超过F0及匹配F1，才值得补训练seed。默认初筛看baseline；是否替代Final还要看完整both与逐导联得失。
同一验证集长期选模带有选择偏差，单次小提升不能证明稳定泛化。

每个评估目录有 `summary.csv`、`pair_metrics.csv`、context来源日志及三路逐导联r/RMSE报告。
重点看V1/V3/V5/V6，并检查V2/V4及肢体导联是否退步。按subject_id分析配对患者，不能把窗口当独立患者。
汇总强制核对：best重评是否复现选模、推理包是否复现候选评分；1e-7是数值容差，不是显著性阈值。

## 8. 独立推理包

`packages/D1_best.pt`包含直接回归、原P、原I头所需FM权重、冻结body及配置，不含优化器、RNG或训练波形缓存。
需要本分支代码，但不需要服务器study缓存或原分散权重文件。

```bash
python -m baselines.B5_directfast.predict \
  --package "$run/packages/D1_best.pt" \
  --anchor-npy /绝对路径/current_I_uV.npy \
  --metadata-json /绝对路径/demographics.json \
  --record-key 当前记录唯一ID \
  --task-id task1 --input-type watch_ecg \
  --output /绝对路径/全新推理输出目录 --device cuda
```

输入I为原始μV、500Hz、完整120秒，形状`[1,60000]`或`[60000]`。
Task2相应指定设备及原始历史和mask：`--history-npy`、`--history-mask-npy`必须同时提供。
不预先对body历史重复滤波，保存的B策略会执行一次。
输出`prediction_uV.npy`为`[12,60000]`，导联顺序写入inference.json。
`--route baseline/d6_only/both`可明确选择修正路由。
F1推理包仍需ODE；D1只做直接前向及固定P/context。

这轮实现的是“从现有最佳权重迁移后直接回归”的可运行对照，未新增公开预训练管线。
如果D1有收益，再单独研究公开阶段直接回归或残差FM；不能把不同预训练目标带来的差异全归因于是否采样。
