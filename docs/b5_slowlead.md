# SlowLead ABC：本地实现与服务器运行

本分支从`B5-1022@69bc40f`创建。ABC模型、准备、训练、复评、汇总与断点恢复入口已实现。本说明取代旧方案中的“尚未实现”状态；旧方案保留为设计记录。

## 1. 三组实际行为

|组|原P|新增胸导联解码|II条件|G120|
|---|---|---|---|---|
|A|全部参数更新|关闭且冻结|无|固定、eval|
|B|全部参数更新|第6块后分出条件解码，更新|无|固定、eval|
|C|全部参数更新|同B，更新|当前P原II路径＋固定G120的II；detach后经可训练编码器|固定、eval|

ABC从同一个initializer独立起步；C不会读取B的best额外续训。三组共用相同数据顺序seed43、20轮预算、record batch2、P学习率1e-5、AdamW/weight decay0.01、梯度裁剪1、每5轮完整验证。B/C新增模块学习率1e-4。

B/C共用六个胸导联embedding和两个共享时间卷积块(dilation64/128)，通道LayerNorm及FiLM调制。条件包含导联身份、当前I的第6块特征摘要及人口学表征。最终1x1输出零初始化，胸导联慢残差接回原P，前五个缺失肢体输出没有直接新增残差。

S50、50Hz、中央10秒裁剪、120秒插值均保留。特征摘要在50秒窗口内池化，新分支也增加容量与计算路径，因此B/A代表该结构整体效果，不是严格排除了容量和感受范围因素的机制证明。

C先以原P路径计算完整120秒II慢预测，加固定G120的II慢修正；在no_grad内构造条件。按与I相同的50秒窗口坐标裁剪II，经两个1x1卷积编码后注入共享特征，再执行导联条件卷积。II编码器可以更新；胸导联损失不会经II条件反向修改II预测路径。不输入真实II或历史II，不依赖胸导联输出，避免循环。C因此比B多一次无梯度P前向。

## 2. 本轮基准与固定历史路由

本轮固定使用原G120实验对应的Final历史路由：机器FIR25、原body权重。代码继承自B5-1022，但不会自动挑选或接入另一次G120-C3实验的权重。原方案里的“保留既有历史处理”在本轮明确指这个固定路由，不能声称已复现任意C3部署组合。

FM不重训、不重新采样，读取源缓存；G120固定；beta=.75。训练监督总慢预测`P_new+G120`，并复用原慢损失与固定FM组合PCC；训练损失不经过context，正式选模与复评经过同一固定context。A不能复用会更新G120的H_add。

prepare使用原G120共享缓存及其best，严格核对原代码指纹、数据QC、权重和归一化。ABC均须在完整真实验证流程复现源成绩，容差2e-6；通过后才写出study.json。源资产与代码不兼容时停止，不跳过指纹核对。

## 3. 本地提交后推送

在本地PowerShell执行：

```powershell
git -C 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\B5-1022-SlowLead' push origin B5-1022-exp-slowlead
```

本地commit不等于已经推送；本轮实现不会自动发布或启动服务器训练。

## 4. 服务器创建独立工作树与slowlead tmux

在服务器终端执行，路径沿用已有`/home/qht/huawei`仓库：

```bash
cd /home/qht/huawei
git fetch origin B5-1022-exp-slowlead
git worktree add --detach /home/qht/huawei_slowlead FETCH_HEAD
tmux new-session -s slowlead
```

若服务器的同一仓库远程名是`gitee`，将fetch的`origin`改为该实际远程名；本地上述push仍按本地origin配置。若`slowlead`已存在，用`tmux attach -t slowlead`进入；不要重复创建或删除已有会话。工作树已存在时先检查它的提交和工作区状态，不使用强制覆盖。

进入tmux后只初始化一次：

```bash
cd /home/qht/huawei_slowlead
conda activate BioFlow
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
python -m unittest discover -s baselines/B5_slowlead/tests -v
bash -n scripts/run_b5_slowlead.sh
```

## 5. 一次准备ABC共享缓存

使用全新结果目录，不覆盖旧实验：

```bash
export slowlead_root=/home/qht/huawei/results/B5-1022-exp-slowlead/abc_v1
slowlead_source=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863
test -f "$slowlead_source/shared/study.json"
test -f "$slowlead_source/G120/best.pt"
mkdir -p "$slowlead_root"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5_slowlead.prepare \
  --source-run "$slowlead_source/shared" \
  --g120-checkpoint "$slowlead_source/G120/best.pt" \
  --output "$slowlead_root/shared" --device cuda \
  2>&1 | tee "$slowlead_root/prepare.log"
```

必须出现`ready: True`。这一阶段会复评ABC第0轮，但不训练、不重跑FM、不复制整套源npz。中断准备后保留目录并改用新结果目录重新准备。

## 6. 同一tmux顺序运行A/B/C

以下使用GPU0，三组顺序运行，不同时争抢一张卡。需要其他卡时统一改`--gpu`。

```bash
bash scripts/run_b5_slowlead.sh --run "$slowlead_root" --group A --gpu 0 && \
bash scripts/run_b5_slowlead.sh --run "$slowlead_root" --group B --gpu 0 && \
bash scripts/run_b5_slowlead.sh --run "$slowlead_root" --group C --gpu 0 && \
python -m baselines.B5_slowlead.summarize --run "$slowlead_root"
```

如首轮先看A/B，可先只运行前两条，再执行：

```bash
python -m baselines.B5_slowlead.summarize --run "$slowlead_root" --groups A B
```

汇总目录必须不存在。先做A/B汇总后补C时，为新汇总指定`--output "$slowlead_root/comparison_ABC"`，保留旧报告。

离开tmux但保留训练：按`Ctrl+b`，松开，再按`d`。重新进入：`tmux attach -t slowlead`。

## 7. 断点续训、日志与结果

原命令再次运行会从对应组的`last.pt`按已完成轮次恢复优化器、数据顺序及随机数状态；不会从其他组恢复。训练已完成则仅补best_r/last的真实checkpoint复评。半轮中断会重做该轮；不支持同时在同一组结果目录启动两个写入进程。

如果中断发生在best先保存、last尚未落盘的窗口，程序会检测best比last更新并停止，要求保留现场检查，不会静默拼接不一致状态。初始化目录存在但无last时也会停止，避免覆盖。

```bash
tail -n 30 "$slowlead_root/logs/A.log"
tail -n 30 "$slowlead_root/logs/B.log"
tail -n 30 "$slowlead_root/logs/C.log"
```

主要结果：

```text
shared/preflight.json
A|B|C/initialization.json
A|B|C/history.jsonl
A|B|C/best_r.pt
A|B|C/last.pt
A|B|C/completed.json
A|B|C/evaluations/best_r/evaluation.json
A|B|C/evaluations/best_r/pair_metrics.csv
A|B|C/evaluations/best_r/slow_diagnostics.csv
comparison/summary.md
comparison/summary.csv
comparison/summary.json
```

best保留第0轮候选，按完整120秒最终平均r选择；不以慢r或RMSE选择。汇总核对checkpoint、study与评价人群指纹。B对A判断内部解码增量，C对B判断II条件增量，三组同时对原G120参考比较。

本地CPU单元测试与真实服务器epoch0预检各有用途；前者不能代替真实验证成绩。当前没有新增训练提分结论。
