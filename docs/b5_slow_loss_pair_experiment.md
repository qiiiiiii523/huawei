# B5-exp C/M/F：三卡配对慢预测实验

这轮只训练S30小CNN，不训练或改动B5 Flow Matching主干。三组读取同一已有S30权重、同一prepared数据；新结果写入独立目录。

| 分支 | 默认组 | 唯一损失改动 | GPU |
|---|---|---|---|
| B5-exp-C | C | 原Huber＋0.1 PCC | 0 |
| B5-exp-M | M | 原损失＋完整120秒记录均值Huber | 1 |
| B5-exp-F | F | 原损失＋目标匹配的较快残差Huber | 2 |

三组实现相同，仅`configs/experiments/b5_exp_loss_group.json`默认组别不同。原CNN48通道、30秒I输入、输出中间10秒，再拼成120秒；人口学、质量mask、尺度不变。三组均从第50轮S30模型权重初始化，忽略其优化器/RNG，重新使用同一seed42、AdamW(lr=1e-4, weight_decay=0.01)、record batch=2，新增20轮、每5轮验证。

M不减掉输出均值、不使用真实均值校正预测。F在50Hz完整慢曲线上计算`H(z)=z-居中51点移动平均(z)`，监督`Huber(H(预测),H(标签))`，不要求所有振荡变零；51点约1.02秒，是起始实验定义。两项只用于训练，推理只有I与人口学。

## 1. 本地提交后推送分支，服务器创建工作树

本地三分支已经提交。将三个分支推送到服务器能够访问的同一个远程仓库；例如本地配置了origin时：

```powershell
git -C C:\Users\Ashley\Desktop\HW\huawei push origin B5-exp-C B5-exp-M B5-exp-F
```

服务器BioFlow环境执行以下一次性创建步骤。以下假设服务器可访问的远程名为`gitee`；如果实际使用origin，将`remote=gitee`改为`remote=origin`。不要切换正在使用的主目录分支。

```bash
cd /home/qht/huawei
remote=gitee
git fetch "$remote"
git worktree add --track -b B5-exp-C /home/qht/huawei_exp_C "$remote/B5-exp-C"
git worktree add --track -b B5-exp-M /home/qht/huawei_exp_M "$remote/B5-exp-M"
git worktree add --track -b B5-exp-F /home/qht/huawei_exp_F "$remote/B5-exp-F"
```

仅在分支/工作树尚不存在时执行`worktree add`；已存在则直接在对应工作树`git pull --ff-only`，不要删除原结果。

本轮基于已提交B5-exp，保留`correlation50_bonus10_target_record_v2`评分：主分50、加分10、设备内目标等权、两设备等权。不会引入主目录后来进行的main评分/QC改动；如果以后调整正式评分，应统一重算所有组。本实验正常导联慢r/诊断使用质量mask，完整组合评分保留现有验证集合。

## 2. 共同预检，仅做一次，不训练

直接读取已有权重和prepared，不重新prepare、不复制/冻结原权重。预检在固定前两条训练记录上计算loss及对预测曲线的梯度量级：令每项辅助梯度约为主损失梯度的10%，系数限制在[0.01,10]。裁剪情况写入报告；这是初始量级匹配，不是最优权重或参数梯度平衡的证明。三组固定使用同一报告，不能分别用验证结果调参。

```bash
cd /home/qht/huawei_exp_C
python -m baselines.B5_exp.check_loss \
  --prepared /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared \
  --init-checkpoint /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt \
  --report /home/qht/huawei/results/B5-exp/slow_loss_v1/shared/preflight.json \
  --device cpu
```

确认`ready=true`、`optimizer_updates=0`、`prediction_shape=[2,11,6000]`；查看weights、weighted_gradient_ratios、weight_clipped。预检只是功能和量级检查，不是性能结果；如果辅助项梯度无效会停止。报告已存在时不重复执行，原三组必须复用同一报告。

## 3. 三个终端分别启动，同时运行

终端1（GPU0，C）：

```bash
cd /home/qht/huawei_exp_C
bash scripts/run_b5_slow_loss.sh --gpu 0
```

终端2（GPU1，M）：

```bash
cd /home/qht/huawei_exp_M
bash scripts/run_b5_slow_loss.sh --gpu 1
```

终端3（GPU2，F）：

```bash
cd /home/qht/huawei_exp_F
bash scripts/run_b5_slow_loss.sh --gpu 2
```

每条脚本依次完成本组训练及三份候选评估。保持终端连接，或在各自tmux会话运行。Windows手动上传脚本后如遇CRLF，先`sed -i 's/\r$//' scripts/run_b5_slow_loss.sh`。中断后在同一工作树/同一GPU执行原命令末尾加`--resume`；恢复最新完整epoch，不能用于改变组别、权重或预算。

输出：`/home/qht/huawei/results/B5-exp/slow_loss_v1/{C,M,F}/`。日志在组目录，权重与评估在`checkpoints/`。

| 文件 | 选择规则 |
|---|---|
| best_slow_r.pt | 正常导联、唯一验证目标、120秒慢r最高（50Hz，与原S30一致） |
| best_score.pt | 固定B5快分量＋新慢曲线的原始组合60分最高 |
| last.pt | 最新完整训练状态，用于精确续跑 |

新增第0轮先评估同一初始化并纳入候选。之后1～20是新增轮次，不是原训练的1～20；如果best仍为0，说明该选模指标下没有超过原S30。每次验证记录慢r/RMSE、记录均值误差、原始r1/r2与两设备RMSE/总分、正常导联快残差、独有目标、波峰幅值大点及10秒拼接边界误差。周期组合评分不再对FM采样，始终复用原Heun16/K16的B5预测。

## 4. 三组都完成后比较

```bash
cd /home/qht/huawei_exp_C
python -m baselines.B5_exp.compare_loss \
  --control /home/qht/huawei/results/B5-exp/slow_loss_v1/C/checkpoints \
  --mean /home/qht/huawei/results/B5-exp/slow_loss_v1/M/checkpoints \
  --residual /home/qht/huawei/results/B5-exp/slow_loss_v1/F/checkpoints \
  --output /home/qht/huawei/results/B5-exp/slow_loss_v1/comparison
```

比较会核对同一初始化、prepared、校准报告、预算、初始分数与当前候选SHA。先比较三组last的同等20轮，再比较同一选择规则的best，并与共同初始化比较。M/F高于C但低于初始化，仅说明减轻退步。除了总分，还检查`wave_record_lead.csv`中V2/V3/V4、`wave_summary.csv`中的快残差和边界误差，以及`plots/*fast_detail.png`固定37.5～42.5秒细节图；新增慢曲线不是严格频带限制，仍可能改变组合波峰。

## 5. 打包报告（不打包权重/数组）

```bash
cd /home/qht/huawei/results/B5-exp
find slow_loss_v1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' -o -name '*.png' \) -print0 > /tmp/b5_cmf_reports.list
tar --null -T /tmp/b5_cmf_reports.list -czf /home/qht/B5_CMF_reports.tar.gz
```

本地仅做合成输入的前向、损失反传、接口和工作流检查，不启动真实数据训练。待服务器结果证实有效后再做S50或FM接入。
