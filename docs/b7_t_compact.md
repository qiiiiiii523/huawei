# 两套约2000万参数的B7-T独立实验

本分支B7-T-compact从B7-T@5cffb8e创建。原B7-T工作树、三份基线配置及用户未提交的脚本留在原处。服务器仍在运行的基线不需要停止、迁移或更新；新分支必须使用另一个服务器工作树。

## 配置

| preset | 参数量 | embed_dims | heads | depths |
|---|---:|---|---|---|
| g6_19m | 19,398,702 | 128/256/384 | 4/8/12 | 2/2/6 |
| w144_g5_21m | 21,259,214 | 144/288/432 | 4/8/12 | 2/2/5 |

每套有public10/public40/huawei40三份配置，位于configs/b7_t_compact/PRESET/。两套都从随机T重新训练P，不加载B5、U7或原1506万T权重。Q/A从本套P的EMA最佳权重初始化，B从本套Q初始化，禁止不同结构混用。

| 阶段 | 输入输出长度 | 训练batch | 更新次数 | 验证安排 | 独立保存last |
|---|---|---:|---:|---|---:|
| P | 真实10秒 | 64 | 10000 | 仅10000步末尾一次，不做初始验证 | 每500步 |
| Q | 40秒 | 32 | 1000 | 仅1000步末尾一次，不做初始验证 | 每500步 |
| A/B | 40秒 | 32 | 各1000 | 仅1000步末尾一次，不做初始验证 | 每500步 |

P累计64万窗口条次，是当前1506万基线32万的两倍。所有阶段均跳过初始和中途全量验证，只在本阶段结束时以最终EMA进行K16评估；best.pt/best_r.pt在最终完整验证后才产生，代表最终EMA，不是在多个中间检查点中选优。训练仍每50步输出轻量loss进度。不把训练window_r作为选best分数。

## 保留的任务规则

完整I直接作为输入第一通道，不加噪、不进入ODE随机状态。七个生成状态为II与V1～V6，网络8入7出，最终复制I并在物理单位下恢复四条肢体导联，得到12导联。FM/MAE使用全窗真实有效位置，40秒PCC只取中央15～25秒；II训练权重5，最终评分不额外乘5。

K16波形平均、Heun16、验证轨迹batch32、固定噪声key、40秒滑窗取中央10秒拼完整记录、原患者split/QC/冻结scale和整记录评分均保留。P/Q按公开验证r，A/B按华为完整120秒的task1/machine/body加权r选best。减少验证不代表模型一定更好；最终r需真实训练后判断。

新增validation.initial布尔字段，省略时仍默认true；新实验所有阶段均显式设false。未验证的last保存best_r=-inf、last_validated_update=-1，这表示没有评估，不是模型生成失败。恢复时不补初始验证；若最终验证中断，则先补完该次验证，不能直接标记completed。每次结束仍强制验证。

## 服务器：创建独立工作树（普通终端）

本地提交由用户人工push到gitee后，在服务器执行。不要在正在运行的/home/qht/huawei_B7_T里checkout或pull新实验代码。fetch和创建另一个worktree不会替换旧工作树里的代码。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T
git fetch gitee
git worktree add --track -b B7-T-compact /home/qht/huawei_B7_T_compact gitee/B7-T-compact
cd /home/qht/huawei_B7_T_compact
python -m unittest discover -s baselines/B7/tests -v
```

上面是首次创建；分支或路径已存在时应使用已有的新实验工作树，不重复add。

## 一键测速后启动（tmux内）

普通终端进入新会话：

```bash
tmux new-session -s b7t-g6-19m
```

然后在这个tmux内执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T_compact
export B7T_PYTHON="$(command -v python)"
export B7T_CACHE_RUN=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42
export B7T_GPU=0
bash scripts/run_b7_t_compact.sh g6_19m
```

另一套使用独立tmux会话和输出目录：

```bash
# 普通终端创建并进入
tmux new-session -s b7t-w144-21m

# 以下在该tmux内执行
conda activate BioFlow
cd /home/qht/huawei_B7_T_compact
export B7T_PYTHON="$(command -v python)"
export B7T_CACHE_RUN=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42
export B7T_GPU=0
bash scripts/run_b7_t_compact.sh w144_g5_21m
```

两条是独立启动示例，不表示建议立即同卡并行。先查看GPU当前空余显存及第一套的实测峰值，再安排另一套。Ctrl+B松开再D离开tmux；Ctrl+C中断本套总控，后续用同一命令恢复。

脚本先按正式配置测10秒batch64、40秒batch32，每个组合3次预热+10次计时，共26次临时更新，不跑公开全量验证，不保存正式模型权重。两组均通过默认35GiB的PyTorch reserved预算后才自动开始训练。OOM或超预算会保存JSON报告并以非零状态退出，脚本停止，不启动正式训练。CPU-only结果也不能通过GPU启动门槛。

默认每套最多一个训练子进程，因此顺序为P→A→Q→B。确需同套A/Q并行，可设B7T_MAX_TRAIN=2；需另外核算两条40秒训练的总显存。单进程预算通过不保证共享GPU上多个进程同时安全。

复用的是旧run中的data/public10、data/ptb40、data/huawei完整manifest和NPZ；不重新prepare，不修改这些数据，也不使用旧权重。不要把B7T_CACHE_RUN设为只有空data目录、实际引用其他缓存的sparse运行；要指向真正保存三套缓存的原始run。原缓存不能删除。

## 单独测速或仅预览

```bash
python -m baselines.B7.compact_probe --preset g6_19m --window-seconds 40 \
  --output results/B7-T-compact/g6_19m/probe40_manual.json

python -m baselines.B7.compact --preset g6_19m \
  --data-root /home/qht/huawei/data --metadata /home/qht/huawei/metadata \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --reuse-data-from /home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42 --dry-run
```

probe报告含seconds_per_update、windows_per_second、peak_allocated_gib和peak_reserved_gib。容量检查包含预热及首次AdamW分配，不只计稳定阶段。计时不含数据读取，不是完整训练/验证耗时预测；GPU上其他进程也会影响速度。PyTorch reserved不含其他进程或全部CUDA上下文开销，需要再留余量。默认35GiB只是排实验门槛，不是强制显存限制或保证真实训练永不超过40GiB。

脚本环境变量还可设B7T_MEMORY_BUDGET_GIB、B7T_OUTPUT、B7T_DATA_ROOT、B7T_METADATA、B7T_PTB_ROOT，以及B7T_GPU_P/A/B。手动测过且决定直接启动时可显式设B7T_PROBE_FIRST=0。不要用这些变量指向旧基线的输出目录。首次probe报告已存在时不会覆盖，脚本自动使用新的时间戳。

自动probe使用B7T_GPU_P；若把后续A/Q/B安排到不同型号或不同显存容量的GPU，应分别在对应GPU上执行单独probe。

## 结果与恢复

默认结果分别在results/B7-T-compact/g6_19m/seed42和results/B7-T-compact/w144_g5_21m/seed42。bootstrap.log记录测速，probes保存JSON，pipeline.log及pipeline_status.json记录调度，logs/P.log等记录训练，P/A/Q/B保存last、best、history及checkpoint_history。

测速结束、正式训练启动后，在普通终端查看：

```bash
B7T_RUN=/home/qht/huawei_B7_T_compact/results/B7-T-compact/g6_19m/seed42
python -m json.tool "$B7T_RUN/pipeline_status.json"
tail -n 30 "$B7T_RUN/logs/P.log"
tail -n 5 "$B7T_RUN/P/checkpoint_history.jsonl"
```

各阶段的history.jsonl在该阶段最终验证成功后才创建，前期没有此文件属于预期行为；训练进度看logs/P.log和checkpoint_history。另一套把路径中的g6_19m替换为w144_g5_21m。

之后中断恢复，重新运行本套同一Bash命令即可；已有P/last.pt时不再自动测速。仍检查代码、选中的配置目录、manifest和逐文件hash。两套各自锁定配置，不会因另一个preset的JSON变化而改变本套指纹；运行中也不能修改共享Python代码。

旧B7-T的reschedule是为807b8b0→5cffb8e的调度迁移设计，明确拒绝本实验代码；不能用它把旧1506万checkpoint迁移为新架构。新实验从零开始，不需要迁移工具。

## 已验证范围

配置中的参数量由实际模型统计。测试覆盖各阶段配置只在末尾验证、实际评估使用最终EMA而非普通权重，以及训练确实只在末尾验证、首轮验证前的恢复点逐状态续训、最终验证中断补评、两套独立调度/指纹、原配置保持不变、非法配置拒绝、GPU探测的CPU控制流。另用两套正式结构完成CPU上的真实40秒长度、batch1、FM/MAE/PCC反向传播、AdamW及EMA更新；参数有限，I保持不变。记录见b7_t_compact_cpu_probe.json。

完整103项测试通过，新增16项compact检查；实际Bash控制流测试使用假Python命令，确认先测两种长度、失败阻止训练、恢复跳过测速。两套真实调度dry-run均为P→(A,Q→B)，复用缓存时没有prepare任务。模型、数据、loss、采样和评分等13份受保护代码及原三份配置与5cffb8e逐字节一致（归一化换行）。范围核查见[b7_t_compact_check.json](b7_t_compact_check.json)。

尚未在服务器测量正式10秒batch64、40秒batch32的显存或吞吐，未运行公开预训练/华为微调，不承诺新模型优于原基线。

本次统一末尾验证仅用于尚未启动的新compact实验。已启动的旧配置不能直接覆盖后续训，严格指纹检查会拒绝配置变化。每500步覆盖保存last.pt（模型、EMA、AdamW、随机状态及步数），不保留每个历史快照；中断可能重做最近保存后不足500步的更新。最终评估中断可从最终恢复点补评，无需重训。
