# B5-exp：S10/S30慢分量预测能力实验

本分支从最新B5提交`db4b29f01936a51e90fea80b8e4f5ecbed164d48`创建，包含10月13日的评分修正。原B5网络、训练损失和推理没有修改。新增代码独立放在`baselines/B5_exp/`，只训练小型慢曲线预测器，尚未接入FM。

## 1. 实验要回答什么

| 设置 | 输入 | 输出 | 其余条件 |
|---|---|---|---|
| S10 | 当前10秒同步I＋人口学 | 当前10秒11个缺失导联慢曲线 | 相同 |
| S30 | 前10秒＋当前10秒＋后10秒同步I＋人口学 | 当前10秒11个缺失导联慢曲线 | 相同 |

两组同网络、同随机初始化、同训练目标、患者划分、导联mask和预算。人口学沿用B5年龄、性别、身高、体重及缺失mask编码。小网络独立从头训练，不加载B5权重；旧B5只提供原数据清单、固定尺度和对照预测。

“慢分量”统一定义为完整120秒原始信号的501点居中移动平均（500Hz下约1秒，两端重复填充）。先过滤整条记录，再裁取标签。没有逐窗口去均值。慢分量不是基线噪声标签，不删除或改写原始目标。

为了降低成本，输入I及慢标签均用固定10点平均后降至50Hz，两组使用同一处理。网络为48通道、8个膨胀残差卷积块，膨胀率1/2/4/8/16/32/64/128，每块两层核宽5卷积、按时间点做通道LayerNorm及人口学FiLM。感受野约41秒，能覆盖30秒输入；无Transformer。输入降采样是这次小探测器的共同设置，不是改动B5的输入预处理，也不能据此判定500Hz条件中没有可预测信息。

每次按完整记录批处理，将12个中心窗口预测拼成120秒，计算保留绝对电压的Huber＋0.1×整条记录PCC损失。坏导联按原QC mask不参与监督。默认50轮、每5轮验证、记录batch=2、AdamW学习率1e-4、seed42。两组CPU初始化SHA应相同。

小网络`best.pt`按独立验证目标、正常导联的120秒慢曲线r（50Hz）选择；`last.pt`用于续训。这是能力诊断指标，不是B5综合分选模。最后评估会上采样整条预测至500Hz，比较与原501点慢曲线的r/RMSE。

## 2. 新评分与旧结果的关系

本分支使用最新B5评分函数：每条配对先拼接120秒，同一物理目标内配对得分取平均，再对目标等权；Task2两设备1:1；主分50×(r1+r2)/2，额外加分最高10，总计60。

可以读取此前B5的best.pt和Heun16/K16评估数组，因为网络、输入条件和实际预测数组没有因评分修正失效。程序检查原训练数据/人口学SHA，不把旧checkpoint用于FM续训。对照和混合波形均重新按最新目标等权规则算分，因此Task1分数可能与旧配对等权报告不同。不要把这种统计变化解释为网络变差。

测试/验证预测器只接收同步I、人口学，不接收真实胸导联、真实慢曲线或真实目标均值。准备阶段使用真实目标生成监督标签，和一般监督训练相同。30秒邻居是当前记录中测试可取得的I，不是历史设备context，也不支持实时因果推理。

## 3. 服务器准备

在已安装PyTorch、NumPy、PyYAML、matplotlib的训练环境执行。沿用服务器已有数据、修正后的人口学文件及原运行结果，不需要重新公开预训练、重拟合scale或改缓存。

```bash
cd /home/qht/huawei
git fetch origin B5-exp
git switch B5-exp
git pull --ff-only origin B5-exp
```

若服务器使用已配置的gitee镜像，将上面的`origin`改为`gitee`，并确保镜像已有该分支。脚本本身不执行fetch、不访问网络。第一次切换远程分支若Git无法自动跟踪：

```bash
git switch --track -c B5-exp origin/B5-exp
```

默认读取以下原运行：

```text
/home/qht/huawei/results/B5/
└── main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/
    ├── best.pt
    ├── resolved_config.json
    └── evaluation_best_resume_20261012_173139_2232152/
        ├── inference_sampling.json
        ├── task1/{prediction_uV.npy,target_uV.npy,window_metadata.csv,overall_metrics.csv}
        └── task2/{prediction_uV.npy,target_uV.npy,window_metadata.csv,overall_metrics.csv}
```

只传过报告而没有NPY不能运行。数据路径从原运行配置读取；SHA检查失败说明原运行使用的文件后来被替换，先核对，不能跳过检查来混用数据。准备阶段会哈希大数组，GPU空闲是正常的，终端会打印当前检查文件。

## 4. 一条命令顺序完成，使用GPU1

```bash
cd /home/qht/huawei
bash scripts/run_b5_slow_probe.sh --gpu 1
```

执行顺序：合成CPU功能检查 → 数据一致性检查/准备 → CPU前向预检 → S10训练/评估 → S30训练/评估 → 配对结果汇总。单卡依次运行。首次运行自动新建带时间戳目录，不覆盖旧B5或旧探测器结果。不会开始完整FM训练。

默认50轮只是首次探索预算，不保证最优。若显存不足，首次启动时同时减小两组记录batch：

```bash
bash scripts/run_b5_slow_probe.sh --gpu 1 --batch-records 1
```

若需手动分步，使用固定新目录，执行如下命令。已有目录中的结果不要覆盖。

```bash
cd /home/qht/huawei
run=results/B5-exp/slow_probe_manual_v1
source_run=results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239
mkdir -p "$run"

python -m baselines.B5_exp.prepare \
  --checkpoint "$source_run/best.pt" \
  --evaluation-dir "$source_run/evaluation_best_resume_20261012_173139_2232152" \
  --output "$run/prepared"

python -m baselines.B5_exp.check \
  --prepared "$run/prepared" \
  --report "$run/preflight.json"

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_exp.train \
  --prepared "$run/prepared" --context-seconds 10 \
  --output "$run/S10" --device cuda \
  --epochs 50 --batch-records 2 --validate-every 5 --execute-training

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_exp.evaluate \
  --prepared "$run/prepared" --checkpoint "$run/S10/best.pt" \
  --output "$run/S10/evaluation_best" --device cuda

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_exp.train \
  --prepared "$run/prepared" --context-seconds 30 \
  --output "$run/S30" --device cuda \
  --epochs 50 --batch-records 2 --validate-every 5 --execute-training

CUDA_VISIBLE_DEVICES=1 python -u -m baselines.B5_exp.evaluate \
  --prepared "$run/prepared" --checkpoint "$run/S30/best.pt" \
  --output "$run/S30/evaluation_best" --device cuda

python -m baselines.B5_exp.compare \
  --s10 "$run/S10/evaluation_best" \
  --s30 "$run/S30/evaluation_best" \
  --output "$run/comparison"
```

## 5. 中断后继续

用首次打印的完整输出目录替换示例目录，其他参数保持首次运行一致：

```bash
bash scripts/run_b5_slow_probe.sh \
  --gpu 1 \
  --output results/B5-exp/slow_probe_实际时间戳 \
  --resume
```

已完成的组跳过训练，未完成的组从自己的last.pt续训。训练状态含优化器/RNG/数据指纹。若评估阶段中断留下非空、不完整目录，先将该组evaluation_best重命名保存，再重跑；不要删除B5结果目录。若训练预算改变导致best.pt刷新，旧评估会被识别为过期，同样另存旧评估目录后重评。

比较汇总如果已存在但对应旧评估，脚本会提示过期；也需另存comparison目录后重跑，避免把新权重与旧汇总混用。

## 6. 看什么结果

输出根目录包含prepared/、preflight.json、S10/、S30/、comparison/。

- `comparison/report.md`：固定B5与两个“预测慢替换”结果的原始r1/r2、两设备胸导联RMSE和总分。
- `comparison/slow_curves.csv`：共同/独有/全部独立目标、全部/正常导联的完整慢曲线r/RMSE。
- `S10或S30/evaluation_best/slow_record_lead.csv`：逐记录逐导联，便于定位V3/V4/V6。
- `plots/`：独有目标中按原V4表现挑选低/中/高三条，画输入I慢曲线、真实/预测慢曲线和原始波形，不只挑最坏案例。
- `history.jsonl`、`resolved_config.json`：训练趋势、预算和相同初始化SHA。

“预测慢替换”=`原B5预测−原B5预测的慢分量＋小网络预测慢曲线`，只改11个缺失导联，原B5的I头保留。它没有用真实目标校正，但还不是把分支加入FM后的模型，不应标成正式新FM成绩。没有直接把慢曲线加到完整B5输出上，避免双算慢分量。

如果S30比S10更好，并且独有目标慢r改善、混合原始r/总分也改善，才值得把长I分支接入FM并做同预算继续训练对照。如果只改善慢指标却损害原始波形，要检查幅度与快慢叠加；若两组都学不好，不应直接证明I没有信息，应先检查探测器容量/降采样/优化，再决定是否需要其他条件。首次固定种子结果是方向判断，不能当成稳定提升结论。

## 7. 打包给我分析

把run改为本次实际输出目录，下面只打包日志、表格、说明、图，不传权重和大数组：

```bash
cd /home/qht/huawei
run=results/B5-exp/slow_probe_实际时间戳
find "$run" -type f \( \
  -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o \
  -name '*.md' -o -name '*.log' -o -name '*.png' \
\) -print0 | tar --null -T - -czf B5_slow_probe_reports.tar.gz
```

下载`/home/qht/huawei/B5_slow_probe_reports.tar.gz`即可。
