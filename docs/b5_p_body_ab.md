# P最佳模型：body原始context A／去漂移context B配对实验

## 1. 这轮做什么

固定完整P模型：

```text
/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt
```

复用已经完成的P验证预测：

```text
/home/qht/huawei/results/B5-best-context/P_O16_four_v1/base_inference
```

只新增生成P的body训练预测，训练集、验证集使用相同完整P模型、Heun16/K16、推理seed42、慢分量β＝0.75。P预测只准备一次，A/B重复使用，不训练FM、P慢CNN或d6。

|组|body历史输入|修正器训练|
|---|---|---|
|当前旧O16|原始context A|旧权重直接迁移到P，本次不训练|
|A|原始context A|匹配P预测，从随机初始化训练30轮|
|B|整记录0.2 Hz去漂移context B|匹配同一P预测，从相同随机初始化训练30轮|

A/B均O16编码（共享历史I尺度；心搏截断16、统计截断8）、seed42、batch16、lr0.0002、原PCC＋残差正则、α＝1；每轮按完整120秒、正常I/II、目标记录等权的II r选best，预算含epoch0恒等候选。两组不是从旧body权重继续训练。唯一实验变量是body历史context预处理。

主体训练样本仍受原QC控制；B可能改变心搏检出和历史特征可用性，变化会单独记录，不悄悄删除目标记录。目标、同步I、P预测、患者划分、QC与d6均不变。B不读取当前隐藏目标，不滤当前I或target。

## 2. 先上传B数据

把本地整个目录：

```text
C:\Users\Ashley\Desktop\HW\task2_body_scale_ablation
```

上传为服务器：

```text
/home/qht/task2_body_scale_ablation/
    body_scale_ablation_manifest.json
    body_scale_context_window_metadata_B.csv
    body_scale_train_context_B.npy
    body_scale_train_context_valid_mask_B.npy
    body_scale_validation_context_B.npy
    body_scale_validation_context_valid_mask_B.npy
    README.md
```

如果服务器同位置已有这一整套文件，不必重复上传。这里是`/home/qht/task2_body_scale_ablation`，不是仓库里的`data/`。文件夹里的Windows来源路径仅是来源记录；读取器使用当前服务器的数据配置和文件哈希检查，不要求存在那些Windows绝对路径。

如果实际放在别处，下面**每一步**都添加同一个`--overlay /实际目录/task2_body_scale_ablation`。原canonical metadata、QC或划分哈希如果不匹配，必须为当前数据重建B，不能修改manifest伪装匹配。

## 3. 服务器更新代码

本地提交推送`B5-best-context`后，在服务器对应工作树执行：

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
git log -1 --oneline

sed -i 's/\r$//' scripts/run_b5_p_body_ab.sh
bash -n scripts/run_b5_p_body_ab.sh
)
```

## 4. 准备共享缓存和CPU预检查，GPU1

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_p_body_ab.sh \
  --step prepare \
  --run /home/qht/huawei/results/B5-best-context/P_body_AB_v1 \
  --overlay /home/qht/task2_body_scale_ablation \
  --gpu 1
)
```

这一阶段需要GPU生成body训练记录的P预测，可能比后面小网络训练耗时；验证预测直接复用。随后准备A/B O16特征，检查上传B数据与原始历史按同协议滤波的有效点和特征完全匹配，再检查同初始化、CPU前向／反向和残差边界。不启动优化器训练。

完成标志：

```bash
cat /home/qht/huawei/results/B5-best-context/P_body_AB_v1/preflight_pair.json
```

必须看到`ready: true`和`same_initial_weights: true`。准备完成后再开两组训练，不能和准备步骤同时运行。

## 5. 两个终端分别训练

终端一：A原始context，GPU1。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_p_body_ab.sh \
  --step train-A \
  --run /home/qht/huawei/results/B5-best-context/P_body_AB_v1 \
  --overlay /home/qht/task2_body_scale_ablation \
  --gpu 1
```

终端二：B去漂移context，GPU2。

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_p_body_ab.sh \
  --step train-B \
  --run /home/qht/huawei/results/B5-best-context/P_body_AB_v1 \
  --overlay /home/qht/task2_body_scale_ablation \
  --gpu 2
```

两组输出分别在`P_body_AB_v1/A/`、`P_body_AB_v1/B/`，均30轮。设备不同不改变配置；相同seed不等于跨设备所有运算绝对逐位一致。

只有一张卡时，可以顺序执行两条命令并都用`--gpu 1`。也可以从全新目录一次完成准备、A训练、B训练和评估：

```bash
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_p_body_ab.sh \
  --step all \
  --run /home/qht/huawei/results/B5-best-context/P_body_AB_serial_v1 \
  --overlay /home/qht/task2_body_scale_ablation \
  --gpu 1
```

三者选一套操作即可，不要同时运行串行all和同目录的分步操作。

## 6. 两组完成后统一评估，GPU1

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_p_body_ab.sh \
  --step evaluate \
  --run /home/qht/huawei/results/B5-best-context/P_body_AB_v1 \
  --overlay /home/qht/task2_body_scale_ablation \
  --gpu 1
)
```

评估会先验证A/B均已完成30轮、权重/初始化/数据指纹一致。每个body按自身best轮次评分，不能使用last代替。

|输出group|完整方案|目的|
|---|---|---|
|baseline|P|无context参考|
|d6_only|P＋d6|固定d6效果|
|old_O16_both|P＋d6＋旧O16|复核当前0.706971候选|
|A_both|P＋d6＋新A|是否适配P后更好|
|B_both|P＋d6＋新B|B比A有无额外收益|

四项关键比较：B−A判断去漂移；A−old判断重新适配P；B−old判断是否值得替换当前组合；逐记录II增减判断收益是否集中。Task1五组应相同，有d6四组的机器r应相同。

```bash
run=/home/qht/huawei/results/B5-best-context/P_body_AB_v1
study=$(cat "$run/study_path.txt")

cat "$study/summary.csv"
cat "$study/body_ab_study.json"
```

## 7. 中断恢复和报告打包

重跑中断那一步时，保留原`--step`、`--run`、`--overlay`和GPU设置，在原命令末尾增加`--resume`。不要重复启动同一训练组的两个进程。共享预测和记录特征按哈希继续，body训练从last恢复优化器和随机状态；不更换数据、不把30改成60。评估中断会保留旧目录并选择新评估目录。

```bash
tar -czf /home/qht/B5_P_body_AB_reports.tar.gz \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-best-context \
  P_body_AB_v1
```

## 8. 若最终采用B

B权重内保存完整历史滤波协议；现有body预测接口读取权重后，对原始可见body整记录执行同一滤波，再提取O16特征。推理只能传原始历史，不能传预滤波的B后再重复处理。滤波按导联的连续有效片段执行，保留缺口和尾段，不把历史慢漂移补到当前目标。d6和P模型不变。

本修正器使用历史记录内的RMS归一化，不使用队友CNN的全数据集body scale；不能把其他编码器的scale文件直接替换进来。本轮不新增scale拟合，原P预处理尺度保持不变。旧body权重没有新字段时仍按原始A处理，不改变历史实验行为。
