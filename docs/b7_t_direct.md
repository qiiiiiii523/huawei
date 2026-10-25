# Q40 / Q75公开权重直接做华为微调

分支：B7-T-direct。基于4c2857c；与原B7-T-120工作树分离，不更改已有结果。

每个模型各六组独立华为微调，另训练一次Q75。两个模型共12组华为训练、2组Q75。
初始化从对应公开EMA开始；不使用A40、B40、A120、B120任何华为权重。

|任务|父权重|训练输入/输出|MAE|PCC|最终推理窗口/保留|
|---|---|---|---|---|---|
|Q40_H120|已有Q40|120/120秒|整120秒|中央90秒|120/120秒|
|Q40_H10|已有Q40|10/10秒|整10秒|整10秒|10/10秒|
|Q40_H40C10|已有Q40|40/40秒|中央10秒|中央10秒|40/中央10秒|
|Q75|已有P|75/75秒|整75秒|整75秒|75/75秒，公开验证|
|Q75_H120|新Q75|120/120秒|整120秒|中央90秒|120/120秒|
|Q75_H10|新Q75|10/10秒|整10秒|整10秒|10/10秒|
|Q75_H40C10|新Q75|40/40秒|中央10秒|中央10秒|40/中央10秒|

FM在所有组都使用完整真实窗口。40秒组边缘不参与额外MAE/PCC，但参与FM；波形输入I不加噪。
共享网络8通道输入、7通道速度输出，II权重5，最终恢复12导联；评分口径不变。
所有华为验证都生成并拼回真实完整120秒，固定QC和分组聚合。

Q75复用已有PTB连续NPZ，仅新建引用清单。至少37500真实采样点才能入选，不补长、不重复波形。
记录、患者、划分、归一化尺度不变；日志`[Q75-data]`记录各划分可用记录/患者数和排除记录。
公开75秒验证按完整75秒块取真实前缀，不对最后不足75秒的尾段评分，不补齐。
Q40与Q75的公开验证分数数据范围不同，不直接用于比较；比较六组华为完整记录r。

统一batch16、每组1000更新（Q75亦1000）、lr=1e-5、checkpoint_every=500。
不做初始/中途全量验证；结束时最终EMA，K16、Heun16，评估轨迹批量4。
不同窗口在固定更新预算下看过的总信号时长不同，此方案用于选择实际效果。

每个模型控制器串行：Q40_H120 → Q40_H10 → Q40_H40C10 → Q75 → 三个Q75华为组。
两个控制器合计最多两个任务（含验证）；不是跨所有旧实验的GPU全局锁。
Q75失败则其三个子任务blocked，其他独立任务继续；状态与错误写入pipeline_status.json。
每个任务配置在运行时生成并冻结到configs/，包括公开父权重stage与SHA256。
suite_spec.json记录源P/Q、模型、代码、数据、GPU、预算。相同命令恢复last.pt中的模型、AdamW、EMA、RNG、步数，
已完成且身份通过校验的任务跳过；代码/数据/配置变化拒绝续训，应使用新输出目录。
中断验证会重做该次验证；最终评估完成前没有completed.json。

## 服务器

先将本地B7-T-direct分支人工push至gitee。在普通终端：

```bash
git -C /home/qht/huawei fetch gitee
git -C /home/qht/huawei worktree add --track -b B7-T-direct /home/qht/huawei_B7_T_direct gitee/B7-T-direct
conda activate BioFlow
cd /home/qht/huawei_B7_T_direct
bash scripts/run_b7t_direct.sh 1
```

启动脚本创建tmux `b7t-direct`，两窗口各管理一个模型。GPU号为物理编号；命令不测速。
已存在同名tmux时拒绝重复启动；恢复前确认旧子进程已停止，结束旧tmux或设置新B7T_SESSION再运行相同命令。
已完成数据准备全部跳过。

默认缓存`/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42`；优先PTB恢复目录，再选data/ptb40/manifest.json。
若实际缓存不同，可在启动前export B7T_PTB=/真实路径/manifest.json、B7T_HUAWEI=/真实路径/manifest.json。
默认公开权重根目录`/home/qht/huawei_B7_T_compact/results/B7-T-compact-suite`；可export B7T_SOURCE修改。

查看：

```bash
cd /home/qht/huawei_B7_T_direct
python scripts/show_b7t_direct.py
tail -n 10 results/B7-T-direct-bs16/g6_19m/seed42/launcher.log
tail -n 10 results/B7-T-direct-bs16/w144_g5_21m/seed42/launcher.log
nvidia-smi -i 1
```

源码与CPU测试完成；未在本地进行正式GPU训练或显存测速。120秒batch16此前两模型同卡已运行过，仍需观察启动时实际空闲显存。
