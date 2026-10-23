# 采样率、融合、原慢CNN长窗口实验

本分支可以一次串联执行，也可以先做rates阶段，查看结果后在同一个suite增加后续阶段。所有模型独立保存，互不叠加训练收益，不自动覆盖当前最佳。原Final、原slow120代码及缓存保持原样。

## 两条实验路线为什么在同一分支

`21396c3`实现的是注意力分支的2.5/5/10Hz对照；其后`eeac9ff`把融合、P尾部微调与长窗口感受野实验合入同一套件。它们共享缓存、训练与评分代码，但科学问题不同。分支名是代码集合的名字，不代表这些实验是同一个模型，也不代表已经有新的真实成绩。“24项合成测试通过”只说明当时的代码检查通过。

现在提供明确的运行选择：

|选项|默认阶段|后续训练组|目的|
|---|---|---|---|
|`--preset rate-fusion`（默认）|rates、fixed、followup|CONT、FUSION_SAME、FUSION|只做注意力采样率与多Hz融合|
|`--preset cnn-rf`|rates、followup|CONT、PTAIL、PCTX、PCTX_RF|CNN尾部、长窗口、感受野对照，不跑融合|
|`--preset all`|全部|六组全部|明确要求一次跑全部实验|

这次将套件默认范围改为rate-fusion，原单Hz脚本`run_b5_slowrate.sh`行为不变。已使用旧版源码启动的suite应继续使用旧commit，不能切换源码后直接resume。

注意：当前CNN路线仍以本轮已完成的单Hz模型为冻结全局搭档，所以`cnn-rf`从新目录启动时会先跑三种采样率。它不是“绕过采样率实验，直接从任意历史G120开始”的独立入口。需要独立比较时为两条路线使用不同suite目录；全部路线则用同一个suite。

可用`--stages rates`只跑第一阶段；随后保持同preset与参数，添加`--resume --stages all`继续。也可以用`--groups`手动指定后续组，但它不能与`--preset`同时传入，避免覆盖用户选择。自定义groups不指定stages时仍只跑rates。

先预览实际运行范围，不创建文件、不读数据、不训练：

```bash
bash scripts/run_b5_slowrate_suite.sh --source-shared /原slow120/run/shared --run /计划的新suite目录 --preset rate-fusion --seeds 43,42 --plan
```

把preset换成cnn-rf即可预览CNN路线。检查输出的`rate_groups`与`followup_groups`；删掉`--plan`才会开始执行。GPU仍通过`--gpu 1`指定。

## 实验内容

|阶段/组|实际做什么|比较对象|
|---|---|---|
|rates：R2p5/R5/R10|完整120秒注意力，2.5/5/10Hz匹配训练|本轮R5|
|fixed|两两及三组修正输出等权平均，context只运行一次|最佳单Hz|
|CONT|源单Hz最佳权重继续训练|源权重|
|FUSION_SAME|源单Hz输出重复三次，训练小融合头|新增融合头的容量控制|
|FUSION|三种Hz输出输入同结构融合头，输出头零初始化|源单Hz、FUSION_SAME|
|PTAIL|原P-S50末两块与输出头微调，注意力固定|源单Hz、CONT|
|PCTX|原P改用长窗口，原感受野不变；微调相同尾部|PTAIL|
|PCTX_RF|同长窗口，在原CNN内部增加一个扩张卷积残差块，微调尾部和新增块|PCTX、PTAIL|

默认源单Hz从同seed的三个实际best复评分数中选择。可用`--reference-group R5`预先固定5Hz源，减少在同验证集上选择方向。这里的源是本轮匹配训练模型，不是任意历史G120最好权重；历史最优仍需另按相同患者/预算复核。

FUSION/FUSION_SAME的融合层都是33→16→11的逐时间点卷积，只有最后一层零初始化。三种源注意力固定，三组输出已对齐到50Hz，因此融合不引入标签或额外可见信息。CONT继续训练源注意力，初始预测与源模型一致；融合及PTAIL也必须初始复现源预测。

P实验确实更新原慢CNN参数。默认只解冻最后两个原残差块和输出头，可用`--tail-blocks 1/2/4`统一设置。原FM、全局注意力和context冻结。训练损失沿用原slow120目标，P组额外加入0.1倍的原P输出偏离约束，系数通过`--p-preservation`明确记录；因此P组与CONT不是纯架构单变量消融，PCTX_RF/PCTX才是本轮感受野核心对照。

## 感受野与窗口

原CNN：输入卷积kernel5，8个块各两层kernel5，dilation为1,2,4,8,16,32,64,128。

```text
原RF = 1 + 4 + 8*(1+2+4+8+16+32+64+128) = 2045点 = 40.90秒
新增dilation=d的一个块：RF增加8*d点
默认d=256：RF=4093点=81.86秒
```

默认`--p-context-seconds 100 --p-extra-dilation 256`，每个100秒窗口输出中央10秒，仍在完整120秒记录上拼接。S100对中央10秒最靠边输出提供至少45秒单侧上下文，足以覆盖约40.9秒单侧理论半径。记录本身首尾仍需端点重复，不能凭更长窗口创造记录外观测。

可选S60、S80、S100、S120，必须同时满足“RF秒数+中央10秒 <= 窗口秒数”。例如S120/d384对应102.34秒RF；d512对应122.82秒RF，120秒窗口不能无边界地支持中央10秒所有点，入口会拒绝该组合。

PCTX与PCTX_RF都会真实计算`P_new-P_original_S50`并加入缓存组合。它们改变原CNN运行方式，初始分数可能已变化；summary同时报告initial、best、last。两组加载相同原P，新增块末卷积为零，训练前两组输出严格相同。不能把initial变化伪装成训练增益，也不能假装这两组初始等于原Final或源G120。

## 一次执行全部阶段

推送本地分支并在服务器对应工作树更新后，在服务器终端执行。以下默认物理GPU1、tmux名slowrate；若已有同名会话，先查看它，不重复启动。

```bash
SLOWRATE_SOURCE=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared
SLOWRATE_SUITE=/home/qht/huawei/results/B5-final-exp-slowrate/suite_$(date +%Y%m%d_%H%M%S)

tmux new-session -d -s slowrate "cd /home/qht/huawei_B5_final_exp_slowrate && PATH=/home/qht/.conda/envs/BioFlow/bin:\$PATH bash scripts/run_b5_slowrate_suite.sh --source-shared $SLOWRATE_SOURCE --run $SLOWRATE_SUITE --gpu 1 --preset all --seeds 43,42 --stages all --rate-epochs 40 --followup-epochs 10; exec bash"

echo "完整结果目录：$SLOWRATE_SUITE"
tmux attach -t slowrate
```

每个seed顺序执行：三组采样率训练和best/last复评→四种固定平均→六组后续训练和best/last复评→完整汇总。两seed会明显增加总运行量；不保证每项都提高分数，也不保证总时长。默认后续每组10轮、每轮验证、batch_records=1；P100窗口成本显著高于小注意力，时间与显存记在history。

脚本通过tee等价的逐行转发同时显示与保存日志，训练进程是顺序等待，不需要轮询。按Ctrl-b后按d脱离tmux，原终端可以继续用。最后的`exec bash`保留tmux，会话存在不代表训练还在进行。

## 分阶段与恢复

先只跑采样率，把上述`--stages all`改为`--stages rates`。之后以完全相同source/run/seeds/预算参数加上`--resume --stages all`，已完成步骤会跳过，后续阶段继续执行。必须保存首次输出的真实suite目录，不能重新用时间戳生成另一个路径。

```bash
cd /home/qht/huawei_B5_final_exp_slowrate
conda activate BioFlow
bash scripts/run_b5_slowrate_suite.sh --source-shared /原slow120/run/shared --run /原suite目录 --gpu 1 --preset all --seeds 43,42 --stages all --rate-epochs 40 --followup-epochs 10 --resume
```

恢复会核对源码、源manifest、训练参数，训练从原last.pt恢复优化器与随机状态，完成的权重复评按SHA跳过。未完成的非训练输出会改名为`*_interrupted_时间戳`保留，然后重新生成；未写出last.pt的训练目录也这样保留。`active.lock`防止同一suite重复运行。改变seed/预算/窗口/源码需要另开suite，不能修改旧study。

查看一次当前进度：

```bash
tmux capture-pane -pt slowrate -S -40
cat /实际suite目录/suite.json
```

最终看`/实际suite目录/suite_summary.md`；各seed有`comparison/summary.md`、`fixed/*/ensemble.json`、`followup/comparison/summary.md`。日志在suite/logs，详细患者与慢尺度指标在各verification中。只想做CNN可设置`--groups CONT,PTAIL,PCTX,PCTX_RF`；只做融合可设置`--groups CONT,FUSION_SAME,FUSION`。这些设置也绑定恢复参数。

## 后续判断

固定平均或FUSION必须超过最佳单Hz才有部署意义。FUSION还应超过同规模FUSION_SAME，才支持多率输入的价值。不同Hz和多模型集成的收益需要区分；可直接运行ensemble命令，将同Hz不同seed等权平均作为参考：

```bash
python -m baselines.B5_slowrate_followup.ensemble --rate-runs /suite/seed43 /suite/seed42 --groups R5 R5 --output /新输出目录 --device cuda
```

PCTX_RF若仅超过PCTX但未超过PTAIL和当前最优，仍不作为升级。初筛后应复核训练seed、逐患者变化，避免以同验证集反复挑选得到的微小差值当作确认结果。这里不自动按患者真值选择分支，也不自动发布任何模型。
