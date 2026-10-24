# 仅减少P验证频率的安全续训

独立脚本`scripts/reschedule_b7_validation.py`不改B7模型包；部署时仅checkout这个脚本，不switch新版分支。支持用户服务器032fc7d模型代码；实际运行时以last checkpoint中的code_sha对照当前模型包，任何不一致均拒绝。

条件：仅P未完成，A/Q/B尚未开始；旧P和总控已停止；当前配置必须与保存的P配置完全一致，尚未手动改动validate_every。脚本在Linux扫描/proc并获取旧pipeline锁，检查是否仍有管理同输出的进程。不要误停其他目录B7-T或B5。

脚本核验manifest、每条prepared数据SHA256、冻结scale、模型代码、完整配置以及best/last一致性。将latest last、best_r及best的副本迁移到NEW目录P，只改副本identity.config.training.validate_every，不修改原始模型/EMA/optimizer/RNG/步数/初始化历史/最佳分数；原checkpoint完全保留。新的identity使原train的严格resume正常工作，不关闭原恢复检查。旧config备份和checkpoint来源hash写入新目录migration.json、original_config.json；原history复制后继续追加。只修改仓库的public10配置validate_every一项，A/Q/B配置保持不变。

迁移不增加或重置更新预算，原batch32、学习率、K16、Heun16与评分不变；原best跨新间隔继续竞争。last=4500时后续验证6000/8000/10000；若已经保存了其他步数，以最近last为准。停机之后未保存的更新会重做。无需初始再验证。

## 操作

1. b7中当前batch32总控窗口Ctrl+C，等命令提示符。不要操作B7-T或B5窗口，也不要删除旧结果。
2. 推送含脚本的新提交到Gitee。普通终端只取脚本：

```bash
cd /home/qht/huawei_B7
git fetch gitee
git checkout gitee/B7 -- scripts/reschedule_b7_validation.py
```

3. b7空闲窗口执行整段。不要提前手动修改JSON、创建新输出目录或更新模型代码：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
set -o pipefail
(
  python -u scripts/reschedule_b7_validation.py \
    --source results/B7/batch32_seed42 \
    --output results/B7/batch32_val2000_seed42 \
    --manifest results/B7/auto_seed42/data/public10/manifest.json \
    --validate-every 2000 &&
  python -u -m baselines.B7.pipeline \
    --data-root /home/qht/huawei/data \
    --metadata /home/qht/huawei/metadata \
    --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
    --reuse-data-from results/B7/auto_seed42 \
    --output results/B7/batch32_val2000_seed42 \
    --gpu-p 0 --gpu-a 0 --gpu-b 0
) 2>&1 | tee -a results/B7/val2000_resume.log
```

脚本检查失败时&&阻止总控启动；新目录存在时拒绝覆盖，需检查migration.json及报错后处理，不能盲目删除。新训练仍是原P的参数、优化器与随机状态续训，并非重新初始化。完成P后仍自动P→(A,Q→B)。后续总控进度写入新目录logs/P.log；旧目录不再运行。

观察：

```bash
tail -n 20 -f /home/qht/huawei_B7/results/B7/batch32_val2000_seed42/logs/P.log
cat /home/qht/huawei_B7/results/B7/batch32_val2000_seed42/pipeline_status.json
```

以后再次中断无需重新迁移，保持代码/配置/数据不变，直接重跑上面pipeline部分恢复。A/Q/B仍各1000次更新、每100次验证；本次只优化P。公共配置变化后不要重启旧batch32_seed42目录，原目录保留作回溯。验证变稀会降低检查点选择密度，不改变既有相关系数口径。

## 验证

3项CPU小模型测试通过：原last/best字节不变，模型/EMA/RNG保留；续训结束模型、EMA与AdamW状态与连续训练逐张量一致；改学习率和改数据均在写新输出之前拒绝。未在用户服务器完成GPU续训验证。
