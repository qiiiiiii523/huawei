# B7-T：减少验证并保留现有训练状态

本次修改针对B7-T@807b8b0的batch32运行。模型仍为15,063,342参数，不继承B5/U7；batch32、学习率、AdamW、EMA、总更新次数、数据划分/QC、I不加噪、通道与loss、K16/Heun16及整记录评分不变。

| 阶段 | 总更新 | 独立保存last | 完整验证 |
|---|---:|---:|---:|
| P：公开10秒 | 10000 | 每500步 | 每2000步 |
| Q：PTBDB40秒 | 1000 | 每500步 | 每1000步 |
| A/B：华为40秒 | 各1000 | 每500步 | 每500步 |

新阶段仍先做初始验证，结束强制保存与验证。从已有last恢复不重新做初始验证；如果恢复点标记某次验证未完成，则先重做该次验证。每500步不是保留一个永久文件，而是原子更新last.pt。best.pt/best_r.pt只由真实完整验证更新。

例如从已验证的4500步续P：5000/5500保存，6000保存并验证，6500/7000/7500保存，8000保存并验证，8500/9000/9500保存，10000保存并验证。实际以停止时最新last的步数为准。

## 切换原理与检查

新增`python -m baselines.B7.reschedule`。它只接受已核对的807b8b0代码来源；比较当前提交与原版本，允许train/pipeline/迁移模块变化，模型、数据、loss、采样、推理和评分代码必须一致。三份配置除验证/保存间隔之外必须与原配置一致。不能借此改变模型大小、batch、学习率、更新预算、K或评分协议。

工具在旧总控停止后获取旧run锁，并在Linux检查旧总控/训练进程；核对三个完整manifest、逐文件SHA和冻结scale，以及各已启动阶段的config、模型、优化器、best/last身份/分数/步数和完成标记。将checkpoint副本写到新目录，只更新副本的代码/调度身份和迁移来源，不重置模型、AdamW、EMA、RNG、步数或历史best。旧文件保持原样；history及验证报告复制到新阶段。新目录migration.json记录来源SHA、原身份、新配置及完成状态。

支持P仍在训练，也支持P结束后A/Q/B已经开始或完成；已完成阶段核验后跳过，未完成阶段从各自last续训。A/Q需要已完成P，B需要已完成Q。若停止时某阶段尚未生成有效last，或旧best与last不一致，工具拒绝迁移并保留原结果，不能通过删除身份检查强行继续。

自动调度增加`--reuse-data-from`，从旧run/data读取原public10、ptb40、huawei缓存，不执行prepare或生成新划分。迁移运行会额外核对migration.json及缓存内容。之后仍执行原严格resume与完成校验；历史best副本更新为同一新身份，即使后续不刷新best，也能通过阶段完成检查并传给后继阶段。

## 服务器普通终端：先停止，再更新

本地提交由用户push到gitee后再执行。不能在旧训练运行时pull，因为运行中的程序还会读取代码/config来核验后续任务。

```bash
# 向原b7-t会话发送中断，让总控终止并等待它的子任务退出。
tmux send-keys -t b7-t C-c
```

等待旧任务退出。工具还有进程/锁检查，未退出会拒绝执行。原逻辑只在验证完成后保存last；停止后最近last以后的更新会重做，不声称保存了未落盘的更新。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T
git pull --ff-only gitee B7-T
python -m unittest discover -s baselines/B7/tests -v
```

测试通过后，在普通终端进入或创建tmux：

```bash
tmux new-session -A -s b7-t
```

## tmux b7-t内：一次迁移并继续自动调度

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T

B7T_OLD=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42
B7T_NEW=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_sparse_seed42
set -o pipefail

python -u -m baselines.B7.reschedule \
  --source "$B7T_OLD" \
  --output "$B7T_NEW" \
&& python -u -m baselines.B7.pipeline \
  --data-root /home/qht/huawei/data \
  --metadata /home/qht/huawei/metadata \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --reuse-data-from "$B7T_OLD" \
  --output "$B7T_NEW" \
  --gpu-p 0 --gpu-a 0 --gpu-b 0 --max-train 2 \
  2>&1 | tee -a "$B7T_NEW/pipeline.log"
```

迁移先完整核验数据，可能需要一段时间；这是读取和hash校验，不是重新prepare。输出`MIGRATED`才会启动后面的总控。新输出必须不存在；失败时不要删除旧权重或旧缓存。部分迁移目录有complete=false时总控会拒绝启动，应先核查错误。

Ctrl+B松开再按D可离开tmux，任务继续。Ctrl+C会中断任务。原run/data仍供新训练使用，不能删除或改写；原checkpoint也留作迁移溯源。

## 之后中断：只重启pipeline，不再迁移

在tmux中使用同样环境、路径和代码，重新运行上面的pipeline命令（从`python -u -m baselines.B7.pipeline`起），变量B7T_OLD/B7T_NEW仍需设置。不要再次运行reschedule，因为新目录已经存在。配置/代码/缓存改变仍会被严格拒绝。

## 查看进度和耗时（普通终端）

```bash
B7T_RUN=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_sparse_seed42
python -m json.tool "$B7T_RUN/pipeline_status.json"
tail -n 40 "$B7T_RUN/logs/P.log"
tail -n 5 "$B7T_RUN/P/checkpoint_history.jsonl"
```

训练每50步输出轻量进度；`[checkpoint]`含保存步数、最近验证步数、pending_validation与实测保存秒数。`[validation] start`表示开始全量生成验证；history.jsonl新增validation_seconds。复制来的旧history可能没有耗时字段。checkpoint保存不产生虚假的验证r，也不将训练window_r当成选best的分数。

## 已测试范围

新增14项小Transformer/临时合成数据测试：三阶段配置、代码升级白名单与脏代码拒绝、非调度配置拒绝、逐状态迁移及原文件SHA/mtime不变、与不中断训练的模型/EMA/AdamW/RNG逐项一致、非验证步保存恢复、最终验证中断后先补评、历史best不刷新时完成检查、best写入后中断恢复、已完成P与运行中Q共同迁移、只检查及重复目标拒绝、活动进程/未知代码/数据/best异常拒绝、不完整迁移/缓存变动拒绝、复用缓存的P→(A,Q→B)依赖、非法保存间隔。

本地完整回归87项通过。未连接服务器、未停止旧任务、未在真实服务器checkpoint执行迁移；实际最新恢复步数、GPU显存和节省时长以服务器运行结果为准。
