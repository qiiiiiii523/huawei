# C/F新增训练seed复核

此改动位于独立B5-exp-seed工作树，来源B5-exp-C。原main、C/F以及正在修改的S50工作树不变。一个seed工作树即可用--group C/F并行运行两组，不必在同一目录切分支。

新增train_loss --seed（默认42，合法范围0～4294967295），同时控制Python/NumPy/Torch随机流和训练记录shuffle。初始化仍严格加载原S30-best50；第二次重置随机流后开始新增训练，不恢复源优化器/RNG。实际seed写入resolved_config、每轮日志、checkpoint.config及completed.json；精确resume禁止更换seed。旧命令不传seed仍为42。

脚本新增--seed和--group。默认seed42仍使用slow_loss_v1；seed43默认输出slow_loss_seed43/C或F，避免覆盖已有结果。两组使用同一份共同预检、同一原S30初始化、相同20轮预算；仅损失不同。

量级预检仍固定前两条训练记录和seed42，不随续训seed重新调系数。代码实现哈希变化，因此旧预检不能直接复用；请用新代码在新目录生成一次，再共同用于C/F。预检只做前向/对预测曲线求导，零优化器更新。本接口用于新运行；已有seed42运行需要续跑时应使用其原代码和预检，不跨实现版本恢复。

## 服务器操作

下面假设本工作树改动已提交并推送，服务器已将B5-exp-seed检出到/home/qht/huawei_exp_seed。不要在正在运行其他任务的主工作目录切换分支。

第一次创建服务器工作树（远程名以gitee为例）：

```bash
cd /home/qht/huawei
git fetch gitee
git worktree add --track -b B5-exp-seed /home/qht/huawei_exp_seed gitee/B5-exp-seed
```

共同预检一次：

```bash
conda activate BioFlow
cd /home/qht/huawei_exp_seed
python -m baselines.B5_exp.check_loss \
  --prepared /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared \
  --init-checkpoint /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt \
  --report /home/qht/huawei/results/B5-exp/slow_loss_seed43/shared/preflight.json \
  --device cpu
```

ready=true后开两个终端（每个终端激活BioFlow）：

GPU0：

```bash
conda activate BioFlow
cd /home/qht/huawei_exp_seed
bash scripts/run_b5_slow_loss.sh --gpu 0 --group C --seed 43
```

GPU1：

```bash
conda activate BioFlow
cd /home/qht/huawei_exp_seed
bash scripts/run_b5_slow_loss.sh --gpu 1 --group F --seed 43
```

各脚本自动完成训练和三份候选评估。中断后相同命令加--resume，保持group/seed/路径不变。默认结果在/home/qht/huawei/results/B5-exp/slow_loss_seed43/{C,F}/checkpoints/；新旧结果分开。

检查对应目录resolved_config.json的seed为43，两组initial_weights_sha256一致，新增第0轮分数一致。对照seed42与43各自的F−C，优先比较同等20轮last，再看相同选模规则的best，不能拿F43只与C42比较。原compare_loss入口需要C/M/F三组，不适用于仅两组，勿伪造M目录运行。

快速查看本次两组last正式评分：

```bash
cat /home/qht/huawei/results/B5-exp/slow_loss_seed43/C/checkpoints/evaluation_last/learned_slow_hybrid/competition_score.md
cat /home/qht/huawei/results/B5-exp/slow_loss_seed43/F/checkpoints/evaluation_last/learned_slow_hybrid/competition_score.md
```

此复核检验的是固定S30权重下，新增20轮的随机流/数据顺序敏感性，不是完整预训练从头的多种子实验。本地只做合成检查，不启动真实训练。
