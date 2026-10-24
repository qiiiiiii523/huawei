# B7-U7-large：容量与 batch 对照

> 已被[M2/M3两组结构实验](b7_m2_m3.md)替代。下文为历史方案；旧启动脚本CLI现已转向M2/M3。

本分支基于 B7 556499b。原 B7 的默认配置保持原样，两组新配置分别位于
`configs/u7_large_bs32` 和 `configs/u7_large_bs64`。模型仍为干净完整 I 直接拼接七通道
FM 状态的 U-Net；输出七通道速度，最终恢复十二导联。没有 B5 权重继承。

两组宽度均为 152/304/608，time_dim、metadata_dim 均为128，四个瓶颈残差块不变。
参数量按层定义计算为15,327,327，GPU启动测试会用真实模型再次严格核验。
相对于用户提供的B7-T参数量15,063,342，相差约1.75%。

| 阶段 | batch32 步数 / 验证间隔 | batch64 步数 / 验证间隔 | 每组处理窗口条次 |
|---|---|---|---|
| P：公开10秒 | 10000 / 5000 | 5000 / 2500 | 320000 |
| Q：PTBDB40秒 | 1000 / 1000 | 500 / 500 | 32000 |
| A、B：华为40秒，各自 | 1000 / 500 | 500 / 250 | 各32000 |

每阶段保留初始验证，训练中的验证分别为P两次、Q一次、A/B各两次，均包含最后一步。
保存恢复检查点与验证分离：batch32每500步保存，batch64每250步保存。
减少验证可能错过中间更好的权重，最终仍用对应阶段验证集的最佳r选择权重，测试集不参与。

P学习率1e-4，Q/A/B为1e-5，两组暂时一致，未随batch翻倍。AdamW、II权重5、FM/MAE/PCC
及评分规则保留。batch64使用EMA衰减0.998001（0.999平方），使每处理相同窗口数的EMA保留率
与batch32一致。学习率和EMA是起始方案，不代表最优；相同样本预算不等于相同优化轨迹，
两组更新次数不同，AdamW累计正则化也不完全相同。

两组均从随机初始化训练自己的P，再各自执行P→A和P→Q→B。只复用准备好的数据缓存。
训练长窗输入/输出40秒，MAE整窗，PCC中央15–25秒。公开P为真实10秒。
推理仍为K16、Heun16步；为给扩大模型留显存余量，验证记录batch=4、ODE计算chunk=16，
不改变采样个数和评分定义，浮点累计顺序可能有微小差别。

## 服务器运行

将本分支推送到gitee后，在服务器创建独立checkout；无需停止旧B7，也不要直接在旧目录切分支。
下面假设gitee remote名称为gitee，先在旧仓库检查`git remote -v`。

```bash
cd /home/qht/huawei_B7
git fetch gitee B7-U7-large
git worktree add -b B7-U7-large /home/qht/huawei_B7_U7_large gitee/B7-U7-large
tmux new -s b7_large
```

进入新tmux后：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_U7_large
set -o pipefail
mkdir -p results/B7-U7-large
python -u scripts/run_b7_large.py \
  --data-root /home/qht/huawei/data \
  --metadata /home/qht/huawei/metadata \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --reuse-data-from /home/qht/huawei_B7/results/B7/auto_seed42 \
  --gpu 0 \
  --output results/B7-U7-large \
  2>&1 | tee -a results/B7-U7-large/launcher.log
```

若remote名称不是gitee，替换fetch和worktree命令中的remote名称。若目录/本地分支已存在，
不要重复worktree add，进入已有的新checkout即可。离开tmux按Ctrl+b，松开，再按d。

总控自动先做batch32的10秒/40秒训练反传和K16生成测试；通过后完成该组全部P/A/Q/B，
随后自动执行batch64的测试和全部阶段。每组内部同时只运行一个训练进程，避免两个40秒
大模型争抢显存。组内先P再A再Q再B，依赖正确；A失败不阻止Q→B，Q失败阻止B。
一组失败会记录状态并继续另一独立组；某组GPU测试失败不会启动该组训练。
未自动杀其他实验、清缓存或降低batch，GPU资源不足时从对应probe日志检查原因。
本地未实测CUDA，不能保证与现有B5/B7-T等进程共存时显存一定足够。建议资源空闲后启动。

同一启动命令可在停止后重跑：已完成阶段由pipeline核验后跳过，有last.pt的阶段恢复，
没有last.pt的半成品会报错，需检查后使用新输出目录；不会偷偷从头覆盖。代码/配置/数据变化
会被身份检查拒绝，续跑时不要修改配置。总控锁防止同一输出重复启动。

查看：

```bash
cat /home/qht/huawei_B7_U7_large/results/B7-U7-large/experiments_status.json
tail -n 30 /home/qht/huawei_B7_U7_large/results/B7-U7-large/launcher.log
tail -n 30 /home/qht/huawei_B7_U7_large/results/B7-U7-large/bs32_pipeline.log
tail -n 30 /home/qht/huawei_B7_U7_large/results/B7-U7-large/bs32_seed42/logs/P.log
nvidia-smi -i 0
```

第二组路径将bs32替换为bs64。各阶段结果与最佳权重位置保存在各组的pipeline_status.json。
可先运行`python scripts/run_b7_large.py --dry-run`只查看完整计划，不启动训练也不创建结果目录。
