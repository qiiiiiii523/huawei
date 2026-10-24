# 新T长窗迁到GPU0，先一个再按显存决定是否两个

独立脚本`scripts/b7t_long_gpu.py`只修改suite调度记录，不修改baselines/B7代码、六份模型配置、P权重、EMA、优化器、数据及评分。P仍记录为GPU1，六个后续A/Q/B调度到GPU0。`max_train`从控制文件动态读取，默认1；提高到2后下一次调度即可生效，不停止正在运行的训练。

## 首次切换

先由用户把本地提交push到gitee。服务器普通终端更新本独立工作树；本次新增的脚本/文档不属于训练代码hash，P可以继续验证：

```bash
cd /home/qht/huawei_B7_T_compact
git pull --ff-only gitee B7-T-compact
tmux new-session -s b7t-gpu0
```

以下在新的tmux内执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T_compact
set -o pipefail
python -u scripts/b7t_long_gpu.py switch \
  --controller-pid 2148214 \
  --output /home/qht/huawei_B7_T_compact/results/B7-T-compact-suite \
  --gpu-p 1 --gpu-long 0 \
  2>&1 | tee -a results/B7-T-compact-suite/long_gpu_launcher.log
```

2148214是本次记录中已经STOP暂停的旧compact_suite总控PID。工具严格核对进程所属用户、工作树、启动模块、输出目录及暂停状态，并使用pidfd固定身份。PID已变更时拒绝，不查找或停止别的总控。不要对旧总控执行CONT。

P仍在验证时，工具每15秒打印`[waiting]`，P子进程继续运行。两组P均成功产生completed、子进程退出后，工具核验P检查点和逐文件数据身份，才结束已暂停且无活动子进程的旧总控、释放它的锁。原tmux可能显示旧总控`Killed`，这只发生在P子进程已成功退出后，不是中断P训练或验证。

若发现已经启动过新T的A/Q/B目录，或P失败、代码/配置/数据路径/计划不符，切换拒绝；不强改指纹。若因核验失败退出，按日志检查，勿删pipeline_spec或last.pt绕过保护。检查点、完成状态和数据核验可能花几分钟，属于读取核验，不是训练。

迁移保留原pipeline_spec备份及long_gpu_migration.json；仅改变40秒任务的GPU字段。随后复用P，按全局并发1启动第一个长窗（g6_19m:A）。若P尚未完成，命令会等待，因此可以现在执行，不需要人工等验证结束。默认不做GPU短测；实际可用显存仍需用户观察。

## 查看长窗训练显存

在另一个普通终端：

```bash
nvidia-smi -i 0
tail -n 15 /home/qht/huawei_B7_T_compact/results/B7-T-compact-suite/logs/g6_19m:A.log
```

新实验关闭初始验证，因此首个长窗会直接开始训练。出现update=50/1000后看PID显存，继续观察包括阶段末K16验证在内的占用；瞬时nvidia-smi不代表精确峰值。GPU0其他用户的显存会变化，别按一次快照把余量用满。

## 确认够用后增加到2

在普通终端执行，不中断当前任务：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T_compact
python scripts/b7t_long_gpu.py limit \
  --output /home/qht/huawei_B7_T_compact/results/B7-T-compact-suite \
  --gpu-p 1 --gpu-long 0 --value 2
```

这会允许另一个新T长窗进入GPU0。设置回value 1只阻止后续新增进程，不会暂停已在运行的两个。此限额只管理本新T总控，不包括原小B7或其他用户任务。总状态/训练日志仍在原suite目录，不重新prepare。

## 中断后重启

迁移成功后用run，不再使用旧compact_suite命令或旧PID恢复：

```bash
python -u scripts/b7t_long_gpu.py run \
  --output /home/qht/huawei_B7_T_compact/results/B7-T-compact-suite \
  --gpu-p 1 --gpu-long 0
```

仍按last.pt及完整身份检查续训；动态并发值会保留。若旧总控已退出、但迁移没有完成，可在确认没有其他控制器运行后用`migrate`子命令核验并迁移，再用`run`。不要手动修改检查点或删除计划。

本地测试覆盖GPU字段限定、计划备份与审计、拒绝配置变化/锁占用、P未完成拒绝迁移、动态1/2名额及调度计划选择。Linux实时进程接管未在服务器执行，本地用模拟PID/信号覆盖等待子进程结束及只结束空闲总控的流程；服务器工具会在发送信号前做上述保护检查。
