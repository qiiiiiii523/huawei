# 全路线降低验证频率并从现有P安全续训

当前3251615参数U7、训练batch32保持；P总10000更新，Q/A/B各1000更新，学习率、K16、Heun16、训练loss和评分全部不变。改P/Q/A/B验证间隔为2000/1000/500/500，新增独立checkpoint_every=500。定期last保留AdamW、EMA、RNG和步数；best仅由真实验证更新，不把保存checkpoint当验证。

每次即将验证前也先保存last，pending_validation=True；验证完成后清除标志并更新last_validated_update。如果验证中断，resume先补这次验证，再继续更新，防止遗漏选模或把未验证终点判为完成。初始/末次验证保留。

## 为什么保持参数量与batch

当前模型已取得持续改善的验证r，不能仅凭显存未满判定容量不足。10秒/40秒batch32服务器GPU冒烟峰值分配约3.88/15.08GiB，说明当前配置可单进程运行，不代表双进程峰值或多实验共享效率已优化。扩大参数量需要另开架构实验；本次只调验证与保存调度，避免混淆基线。

## 一次迁移与继续自动路线

旧总控仍在P时，在b7中该batch32总控窗口Ctrl+C并等提示符；不要动B5/B7-T。可在普通终端检查pgrep -af 'baselines\.B7\.(pipeline|train)'并依据--output区分。脚本还检查/proc和旧pipeline文件锁，若原目录任务仍存活会拒绝。

先推送最新本地提交到Gitee，再在停止后的U7仓库更新整个版本（与旧仅P脚本不同）：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
git fetch gitee
git switch --detach gitee/B7
```

当前新配置只能配合下面迁移工具，不要直接对旧目录--resume。保持旧目录和全部缓存不变，新目录不要预先mkdir。

在b7空闲窗口复制整段：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
set -o pipefail
(
  python -u scripts/reschedule_b7_all_stages.py \
    --source results/B7/batch32_seed42 \
    --output results/B7/batch32_sparse_seed42 \
    --manifest results/B7/auto_seed42/data/public10/manifest.json &&
  python -u -m baselines.B7.pipeline \
    --data-root /home/qht/huawei/data \
    --metadata /home/qht/huawei/metadata \
    --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
    --reuse-data-from results/B7/auto_seed42 \
    --output results/B7/batch32_sparse_seed42 \
    --gpu-p 0 --gpu-a 0 --gpu-b 0
) 2>&1 | tee -a results/B7/sparse_resume.log
```

如果之前已用9cc3773迁移并运行P，则source改成实际活动的batch32_val2000_seed42；不要从更旧目录回退。支持已知032fc7d/dfd2fc3/9cc3773源码hash；仅接受模型包变化为已提交train.py保存调度及prepare.py既定读取修复，其他计算文件必须逐字一致，不能用此工具更新网络或loss。当前工作模型包还必须与HEAD一致。未识别版本、数据改变、学习率/batch/预算改变、best与last不一致、A/Q/B已经启动均拒绝。

迁移不修改旧checkpoint；复制last/best及历史到新目录，仅更新新副本的调度配置和经核验的代码身份，保留来源hash与原配置备份。重新构造model/optimizer后由原train严格resume加载完整状态，不是init-checkpoint，不重置优化器。核验所有缓存文件hash可能耗时，不代表训练卡住。

MIGRATED输出实际resume_update；例如5000则后续P验证6000/8000/10000，每500步last。旧5000以后未保存的更新重做。P完后自动A/Q并行、Q结束启动B。数据全部来自原成功manifest，不再prepare；不需要旧PTB恢复进程。

```bash
tail -n 20 -f /home/qht/huawei_B7/results/B7/batch32_sparse_seed42/logs/P.log
cat /home/qht/huawei_B7/results/B7/batch32_sparse_seed42/pipeline_status.json
```

日志包含[resume] update=... validate_every=2000 checkpoint_every=500、[checkpoint]、[validation]逐批进度。暂停后仅损失最近last之后尚未保存更新。以后恢复新目录不再运行迁移工具，直接运行pipeline部分；不要再次创建迁移目录。

## 验证范围

CPU小模型回归覆盖严格续训状态保留、连续训练参数/EMA/AdamW一致、保存频率与验证解耦、验证中断重试、禁止模型计算代码变更及数据/学习率变更。实际服务器GPU续训仍需用户执行验证。减少验证使best候选更稀疏，不能保证与旧频率选出相同best。
