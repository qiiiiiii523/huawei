# B5-exp-S50：只扩大同步I输入范围

分支从B5-exp-C提交c781688创建，沿用B5-exp的SlowCNN、原损失、续训和评估流程。桌面main、C/M/F分支与原结果均不修改。

## 这次比较什么

| 项目 | 已有C/S30对照 | S50实验 |
|---|---|---|
| 初始化 | 原S30第50轮best.pt | 同一个文件；不是C20或F20权重 |
| 输入同步I | 前10秒＋当前10秒＋后10秒 | 前20秒＋当前10秒＋后20秒 |
| 输出 | 中间10秒，12窗拼120秒 | 相同 |
| 网络/人口学/质量mask | 原48通道CNN与已有条件 | 不变 |
| 损失 | Huber＋0.1全记录PCC | 相同，不加M/F |
| 优化 | 新AdamW，lr1e-4，wd0.01，seed42，record batch2 | 相同 |
| 预算/验证 | 新增20轮，每5轮验证 | 相同 |
| 评分 | target_record_v2，完整原始波形60分 | 相同 |

prepared保存的是完整120秒I，因此不重新prepare。记录两端缺失的前后I只重复该记录端点，不跨患者、不读取真实其他导联。50Hz下S50输入2500点，裁取[1000:1500]得到中心500点；拼接输出仍为[记录数,11,6000]，再恢复500Hz。

网络参数仍为213511，名义感受野约40.9秒，不代表每个输出都利用完整50秒。相邻I是当前测试记录可见的同步输入，不是历史设备context。最终输出组合仍使用固定原B5的Heun16/K16预测，不在本轮重新训练FM。

## 服务器操作

先把本地B5-exp-S50推送到服务器可访问的同一个远程仓库。若本地使用origin：

```powershell
git -C C:\Users\Ashley\Desktop\HW\huawei push origin B5-exp-S50
```

服务器默认远程名为gitee；若实际使用origin，替换下面gitee。第一次创建工作树时执行：

```bash
conda activate BioFlow
cd /home/qht/huawei
git fetch gitee
git worktree add --track -b B5-exp-S50 /home/qht/huawei_exp_S50 gitee/B5-exp-S50
```

已存在工作树时只进入`/home/qht/huawei_exp_S50`执行`git pull --ff-only`，不要重复创建，不切换主仓库当前分支。

然后一条命令在GPU2上依次完成预检、训练、三份候选评估、S30/S50比较：

```bash
cd /home/qht/huawei_exp_S50
bash scripts/run_b5_s50.sh --gpu 2
```

脚本默认路径：

```text
初始化：/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt
数据：  /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
对照：  /home/qht/huawei/results/B5-exp/slow_loss_v1/C/checkpoints
新结果：/home/qht/huawei/results/B5-exp/slow_context_s50_v1/
         shared/preflight.json
         S50/train.log
         S50/checkpoints/best_slow_r.pt
         S50/checkpoints/best_score.pt
         S50/checkpoints/last.pt
         S50/checkpoints/evaluation_{best_slow_r,best_score,last}/
         comparison/report.md
```

本分支推荐入口`run_b5_s50.sh`默认S50；底层`train_loss`默认S30用于保留旧接口。日志group=C表示使用原损失，不表示输入仍为30秒；检查`context_seconds=50`。

### 预检会验证什么

预检不训练，检查已有C确实完成相同20轮、初始化/数据/原代码版本与本轮一致，评估与checkpoint哈希匹配。使用同一模型对一条训练记录比较旧/新S30切片和完整预测，要求逐值相同；核对CNN计算定义、损失、评分与B5运行时未变，并验证S50输入/输出形状。通过后才复用已有C。

预期最后输出`ready=true`、`s30_bitwise_equal=true`、`s30_max_abs_difference=0`、`optimizer_updates=0`、S50输入`[12,1,2500]`和拼接输出`[1,11,6000]`。预检检查的是接口及兼容性，不是模型性能，也不是完整GPU训练的逐步复现证明。

新实现有新代码哈希，所以在新结果目录生成新的预检报告；不复用旧CMF的preflight.json。辅助权重在新报告中只是保留通用训练接口所需的元数据，本轮M/F系数始终为0。

如果C的初始化、评分或预算已被改动，预检会拒绝，不能忽略错误把它当公平对照。不要将C/F新增20轮权重当成本轮初始化。两组初始化权重相同，但输入范围不同，第0轮预测与评分可以不同；比较报告会分别记录。

### 中断续跑

保持终端连接，或在tmux中运行。中断后同一目录执行：

```bash
cd /home/qht/huawei_exp_S50
bash scripts/run_b5_s50.sh --gpu 2 --resume
```

不修改预算或上下文；脚本从本组last恢复并继续评估。旧S30/S50输出都不覆盖。

## 结果怎么看

先比较相同20轮的last，再比较相同选模规则的best，并检查正常导联慢r/RMSE、完整原始r1/r2、两设备RMSE/总分、独有目标、V2/V3/V4快细节和窗口边界。原C对照为总分35.199；原F结果不是此次S30原损失对照。

若S50有效，再单独测试与F组合；若收益有限，再固定S50验证更大感受野，不同时加Transformer。当前不做MF、S50+F、更多seed或真正FM接入。

## 打包报告

```bash
cd /home/qht/huawei/results/B5-exp
find slow_context_s50_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' -o -name '*.png' \) \
  -print0 > /tmp/b5_s50_reports.list
tar --null -T /tmp/b5_s50_reports.list -czf /home/qht/B5_S50_reports.tar.gz
```

下载`/home/qht/B5_S50_reports.tar.gz`。对照摘要会写入comparison，已有CMF完整报告可沿用；不需打包大数组与权重。

## 本地功能检查

32项CPU合成检查通过，覆盖S50切片/端点填充/患者隔离/中心裁剪/梯度、旧S10/S30逐值兼容、网络定义与参数量、预检与对照哈希、训练入口第0轮保存、精确恢复、禁止切换上下文续跑、最终评分与比较流程，以及原B5评分与120秒输入规则。Bash语法及git diff格式检查通过。

测试禁止AdamW.step，训练迭代器在第一个训练batch前停止，仅执行合成前向与损失求导；比较工作流的20轮完成信息为测试夹具，不是训练结果。本地没有启动真实训练，没有验证S50的性能收益；收益需服务器实测。
