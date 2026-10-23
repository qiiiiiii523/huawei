# ContextPublic：完整两阶段实验操作

分支`B5-final-exp-context-public`。复用V6结构，不扫描注入位置；只新增“公开真实跨记录配对→华为机器适配”的流程。

## 当前状态

已实现prepare、references、train、evaluate、compare和机器新记录predict。
不是已经训练完成：需要服务器具备源资产，通过数据QC、前向测试和完整基准复评后才能启动优化器。
旧`b5_context_public_review.md`记录的是审计阶段历史状态，以本文件为准。

- 原FM训练目标/独立高斯噪声不改；验证/选模/推理使用Heun16、8对正负噪声K16、scale1、seed42。
- control与context_public相同公开目标、华为目标与更新预算；每个物理目标每轮曝光一次，随机取一个真实10秒窗。不是旧训练里一轮遍历12个窗的100轮预算。
- 默认公开20轮→华为20轮，record batch2，LR1e-4、weight decay.01、EMA.999、seed43，每5个华为轮次完整验证。
- 公开阶段固定预算last EMA进入华为阶段，重置优化器。公开验证患者不参与梯度；本版不使用公开验证成绩挑选轮次。
- 只替换机器FM路由。Task1/体脂秤继承指定完整G120报告的分项分数；不会用新FM重跑这两条路由。
- 冻结P/G120/beta.75和末端FIR25。机器验证重新从噪声生成，不复用旧独立噪声快残差。
- 内部正确/打乱/禁用历史不改变末端正确历史。新记录入口只处理机器路线；Task1/体脂秤仍调用原稳定入口。

## 1. 服务器创建工作树及tmux

推送本轮新提交后，在服务器执行（如仓库使用gitee，将origin替换为gitee）：

```bash
cd /home/qht/huawei
git fetch origin B5-final-exp-context-public
```

首次创建：

```bash
git worktree add --detach /home/qht/huawei_context_public origin/B5-final-exp-context-public
```

如果这个worktree已经创建且没有本地修改，只更新：

```bash
cd /home/qht/huawei_context_public
git switch --detach origin/B5-final-exp-context-public
```

```bash
tmux new -s b5_context_public
cd /home/qht/huawei_context_public
conda activate BioFlow
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest discover -s baselines/B5_context_public/tests -v
```

这会运行合成前向、梯度、全无效历史回退、成对噪声和断点续训测试，不使用真实患者训练。
依赖使用BioFlow已有torch/numpy/scipy/pyyaml/wfdb。测试有失败时先停止，不启动真实训练。

## 2. 一次性准备（例如GPU0）

以下变量在同一tmux终端设置。公开FM必须指定你真实保存的公开训练best.pt，不能填写华为微调权重。
程序检查stage=public、网络配置及冻结scales与当前Final完全兼容，失败时不偷偷替换尺度。

```bash
run=/home/qht/huawei/results/B5-context-public/public_pair_v1
slow_root=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863
public_fm=/填写实际公开预训练运行目录/best.pt
stable_eval=/填写该G120已经完成的完整评估目录
ptb=/home/qht/huawei/data/PTB_db
```

`stable_eval`需要根目录evaluation.json，summary.group=G120，epoch与下方G120/best.pt一致，population与shared一致。
它用于固定Task1/体脂秤原路由；机器参考会用新成对噪声重新计算。
如果PTB在`data/PTB-Diagnostic`，只修改ptb变量。不要新建假目录或换到PTB-XL。
配置、人口学和数据路径从原G120 study读取；需要原结果和配置文件继续存在，不复制一遍全部FM生成缓存。

```bash
bash scripts/run_b5_context_public.sh prepare "$run" 0 \
  --public-checkpoint "$public_fm" \
  --public-root "$ptb" \
  --slow-run "$slow_root/shared" \
  --slow-checkpoint "$slow_root/G120/best.pt" \
  --stable-evaluation "$stable_eval"
```

prepare拒绝覆盖shared。中断后保留旧run，使用新的run路径重做prepare；不要手工将completed改成true。
产物：公共文件完整性审计、单位标定/技术QC、QC后患者配对、Huawei训练/验证缓存、固定P/G120慢预测、源SHA、preflight.json和study.json。
公开split保持原患者哈希划分；不同物理记录一目标一历史；QC后默认至少20训练/5验证患者。训练华为目标去重，验证保持原完整机器目标和QC。

## 3. 成对基准复评（完成后才能训练）

```bash
bash scripts/run_b5_context_public.sh references "$run" 0
```

重新生成公开FM初始、control初始、零初始化CTX初始、稳定华为FM机器路由的完整120秒预测；都接相同P/G120和FIR25。
公开源与稳定华为权重分数不要求一致；control/CTX初始应在2e-6容差内一致，否则停止。
报告带采样协议、实际模型SHA、人群/记录键/scale/慢资产/末端规则指纹。
旧独立K16报告不能通过校验。

## 4. 两组训练（准备只做一次，可用两张卡并行）

终端A：

```bash
cd /home/qht/huawei_context_public
conda activate BioFlow
run=/home/qht/huawei/results/B5-context-public/public_pair_v1
bash scripts/run_b5_context_public.sh control "$run" 0
```

终端B：另建tmux会话后执行：

```bash
tmux new -s b5_context_public_ctx
cd /home/qht/huawei_context_public
conda activate BioFlow
run=/home/qht/huawei/results/B5-context-public/public_pair_v1
bash scripts/run_b5_context_public.sh context_public "$run" 1
```

每组自动公开训练→华为训练→best_r/last完整评估；context组额外做best_r内部打乱和禁用复评。
GPU可以按空闲情况替换。不要让两个进程同时运行同一组。

中断后重新执行对应组的同一条脚本命令，自动从该组last.pt恢复完整epoch，优化器/EMA/RNG/采样和源指纹一并恢复。
中途未保存的epoch会重新计算。数据/源码变更后不能原地续跑，必须新run。
best_r和last保存第0轮候选；所有分项写入history.jsonl。

## 5. 汇总与打包

```bash
bash scripts/run_b5_context_public.sh compare "$run" 0
cat "$run/comparison/summary.md"
```

升级需要CTX-public机器r同时超过control和stable_paired。最终R=0.5*Task1+0.25*machine+0.25*body。
不能把只超过旧独立噪声成绩当作context提升；正确/打乱/禁用均在同一best_r权重上测试。

```bash
cd /home/qht/huawei
find results/B5-context-public/public_pair_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \) \
  -print0 | tar --null -T - -czf B5_context_public_reports.tar.gz
```

## 6. 新记录机器推理

```bash
python -m baselines.B5_context_public.predict \
  --run "$run/shared" \
  --checkpoint "$run/context_public/best_r.pt" \
  --anchor-npy /实际新记录/I_uV.npy \
  --metadata-json /实际新记录/metadata.json \
  --history-npy /实际新记录/history_d6_uV.npy \
  --history-mask-npy /实际新记录/history_valid.npy \
  --record-key 实际稳定记录ID \
  --output /实际新输出目录 --device cuda
```

I为原始μV、500Hz、120秒；历史六导联顺序I/II/III/aVR/aVL/aVF，历史不与当前同步，mask同shape。
缺失历史可同时省略历史两个参数，内部回退无历史，末端回退仅投影。真实目标不作为推理输入。
保留shared/study.json及它指向的模型源资产；无需训练缓存npz参与推理。