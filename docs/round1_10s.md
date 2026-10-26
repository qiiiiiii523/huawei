# 第一轮：M7-L 的 C0～C4，公开10秒 → 华为10秒

本分支 `B7-C-round1` 集中保存五组配置，便于保证实现和数据口径一致。
原 B7-P-L、LF、MS-LF、LH 工作树不改。每组有独立 P/A 输出目录、日志和恢复身份。
本入口不包含 PTBDB、Q/B 或40秒阶段，也不自动提交服务器任务。

## 固定配置

主干为 M7-L：64/128/256/256/256/256/256，7尺度、6次下采样；瓶颈
dilation1/2/4/8/16/32。精确参数量 **14,502,815**。主干、输出头均不改。
完整干净I为条件；生成II、V1～V6七条速度；其余肢体导联由物理单位下的固定公式恢复。
500Hz、10秒、5000点；MAE保持7条，FM/MAE的II权重5，PCC采用缺失11条等权。
所有配置均不增加LF项。已有LH/MS候选留在第二轮，不与第一轮loss同时更改。

|组|PCC11系数|额外项|额外项系数|
|---|---:|---|---:|
|C0|0.1|无|0|
|C1|0.3|无|0|
|C2|0.1|III/aVF目标幅度归一化波形误差|0.1|
|C3|0.3|与C2相同|0.1|
|C4|0.1|II相对I的投影成分误差|0.1|

额外系数0.1是**首轮固定候选值，不是已做服务器梯度标定的最优值**。20µV幅度下限
是预先声明的稳健性设置，不由验证集选取。可用下方训练集诊断记录损失/梯度量级；
不会自动改权重、重跑或停止训练。需要调整时须同步修改实验定义与配置，并使用新实验目录。

### C2

对实际测量的III/aVF标签，以逐记录有效采样点计算总体标准差：

`mean_{eligible record,lead}(MAE(pred,true) / max(std(true),20µV))`。

保留DC误差；排除固定QC无效、非变化目标和padding。先按各导联真实物理尺度恢复
波形，不能先在不同的归一化尺度上做导联相减。该项通过公式仅向II回传梯度。
原FM7、MAE7保持不变。它改变样本/导联的误差权重，不增加新的观测信息。

### C4

用同一个有效时间掩码去均值，设输入I为a、预测II为p、真实II为y：

`delta_alpha = sum((p-y)*a) / sum(a*a)`，
`L_projection = abs(delta_alpha) * std(a) / max(std(y),20µV)`。

输入I标准差低于20µV、I或II固定QC无效时跳过。系数目标来自训练标签；部署时不
需要真实II。不要求II一律像或不像I，也不单独生成I、III或aVF。该项只强调I方向的
成分误差，不能独自解决II的其他形态错误；它不是简单重复一次原始II MAE。

## 训练预算及选择

|阶段|初始化|batch|lr|最大更新|EMA全量验证节点|恢复checkpoint|
|---|---|---:|---:|---:|---|---|
|P：PTB-XL＋Chapman真实10s|同seed42随机初始化|64|1e-4|20000|10000、20000；跳过0|每500|
|A：华为真实10s|本组P最佳EMA；重置优化器|32|1e-5|4000|0、1000、2000、3000、4000|每250|

各组均沿用本组loss到A阶段，比较完整方案。华为连续3次验证未超过停滞基准至少
1e-4、且已训练1000步后早停；4000是上限。best严格保留实际最高r，即使小幅改善
未达到早停delta也仍保存。best可以是A的第0步，表示微调没有超过其公开初始化。
验证使用EMA、Heun16、K16波形平均、ODE batch16、record batch4，不平均r。
公共数据按数据来源等权；华为保留 task1=.5、machine=.25、body=.25 的导联/目标/配对
聚合。华为目标现在是**固定非重叠10秒窗口**，不是旧120秒整体Pearson。
五组共用同一新索引；新10秒成绩不直接与旧40秒上下文/120秒成绩比较。

## 数据：复用uV缓存，只生成一次共享切窗索引

先在服务器新工作树中选择真实存在的源manifest（下列是历史路径示例，必须核对）：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_C_round1
PUBLIC=/home/qht/huawei/data_cache/B7_public_qcv2/data/public10/manifest.json
HUAWEI_SOURCE=/home/qht/huawei_B7/results/B7/auto_seed42/data/huawei/manifest.json
HUAWEI10=/home/qht/huawei/data_cache/B7_C_round1/huawei10/manifest.json
test -f "$PUBLIC" && test -f "$HUAWEI_SOURCE"
python -m baselines.B7.huawei10 \
  --source-manifest "$HUAWEI_SOURCE" --public-manifest "$PUBLIC" \
  --output "$HUAWEI10"
```

该操作不写新波形、不重新拟合尺度：uV原缓存可按新QC-v2公开训练尺度归一化。
保留来源/患者划分/配对/原QC，按0:5000、5000:10000……同时切I及12导联真实标签。
不padding，不将40秒片段误标成10秒；不重分患者。目标窗口ID与配对窗口ID分别保留
父ID+起点，同一目标的多配对仍聚合。窗口中确切恒定目标导联进一步屏蔽，恒定I窗口
标为不可靠，不启用原来排除的导联。继承原已准备数据的同步关系，不声称重新验证了
原始XML的临床对齐。原完整记录应位于稳定路径；不能删除或移动。

索引路径已存在时拒绝覆盖；确认源数据/原尺度信息后可复用已完成索引。
训练入口核对源manifest、公开manifest、所有缓存文件SHA以及配置/代码身份。
重复配对的目标窗口波形或QC不一致时，在开训前报错。

## 单组命令：P成功后自动A，重新执行同一命令可恢复

```bash
python -u -m baselines.B7.round1 \
  --variant C0 --public-manifest "$PUBLIC" --huawei-manifest "$HUAWEI10" \
  --output results/round1_C0_seed42 --gpu-p 0 --gpu-a 0
```

其余组只需将C0及输出目录改为C1/C2/C3/C4。GPU参数接受UUID，优先使用UUID；
两个阶段允许使用不同卡，之后改变卡不改变实验身份。没有添加探测显存停训阈值，
也没有自定并发限制。**不要未经资源安排就同时启动五组**；服务器调度提交和GPU
分配留待实际空闲/预算核对，本次未启动任何训练。

可先加 `--check-only`：检查配置、数据、视图与全量身份，不初始化模型或更新权重。
运行时每组保留 `status.json`、`specification.json`、`logs/P.log`、`logs/A.log`、
`P/best_r.pt`、`A/best_r.pt`、各阶段history/validation/last/completed。
validation文件中逐记录的 `limb_diagnostics` 保存II/III/aVF幅度比、归一化波形误差、
I/II相关性和投影系数；不参与正式评分，也不用于推理校正。
只有退出成功且completed、best/last与身份核验一致才释放下游；P失败不启动A。
中断仅保留最近checkpoint，未保存步数重做；入口会结束自己管理的子训练进程。
同一组并发启动由操作系统锁阻止；配置/数据/代码变化拒绝在原目录恢复。
这些约束不允许把旧分支输出作为本分支的resume目录。

已完成各组可用 `python -m baselines.B7.round1_summary --root results` 汇总P/A的最佳步数、
总r及II/III/aVF的同口径r；只读已保存验证结果，不再次推理。

## 可选训练集loss量级诊断（不是显存探针）

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.B7.loss_probe \
  --manifest "$PUBLIC" --config configs/round1/C2/P.json \
  --output results/C2_P_loss_audit.json --device cuda
```

同理检查C4。A阶段使用对应A配置、华为索引及 `--init-checkpoint 本组P/best_r.pt`。
仅取训练集、batch2、两批，无优化器更新，记录各项loss及加权新项相对FM+MAE+PCC的
**速度输出梯度范数比**，不是整个网络参数梯度；不会自动改正式batch/系数或采样。

## 本地检查

新测试覆盖物理尺度、C2与C4的II梯度、正交残差、低幅度I、无效NaN、padding、原
FM/MAE不变、10秒视图与尺度复用、患者/配对身份、Huawei组权重、验证节点、EMA
跨阶段初始化及优化器重置、早停恢复。另运行原B7回归及真实M7尺寸梯度检查。
CPU结果不等于服务器已验证吞吐、峰值显存或分数提升。
