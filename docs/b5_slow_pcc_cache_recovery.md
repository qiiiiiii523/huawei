# SlowPCC：156条训练缓存完成后，验证缓存报错的恢复

适用报错：`Repeated validation target has different FM ensemble`，随后训练报缺少`cache/cache.json`。

156条训练预测已经保存；失败发生在验证缓存整理，训练尚未开始。旧版错误地要求同一目标在不同原验证配对中的历史预测逐点近乎相同。仅凭这个异常不能认定差异来自数值误差，也不能认定数据或训练模型有问题。

修复版按`任务＋pair_id`保存每个原验证配对的FM预测，保留同一目标的I、人口学、标签、QC一致性检查。评分继续使用原目标记录/导联QC/Task2设备权重，不更改r公式。慢CNN仍按唯一目标预测，正式评分时与每个配对原有FM预测相加。

## 1．更新服务器代码

将本地`B5-best-exp-SlowPCC`修复提交推送后，在服务器执行：

```bash
(
set -e
cd /home/qht/huawei_B5_best_exp_slow_pcc
git fetch gitee B5-best-exp-SlowPCC
git switch --detach FETCH_HEAD
)
```

手动上传也可以，必须同时替换以下5个文件，目录仍为服务器worktree中的`baselines/B5_slow_pcc/`：

```text
__init__.py
common.py
cache.py
evaluate.py
repair_cache.py
```

这些修复不改变网络、损失、训练预算、权重或β0.75/Heun16/K16。

## 2．运行专用恢复入口：CPU即可，无需重算FM

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

python -u -m baselines.B5_slow_pcc.repair_cache \
  --study "$run/shared/study.json" \
  --cache "$run/cache" \
  --finish-cache \
  2>&1 | tee "$run/cache_repair.log"

cat "$run/cache/repair_report.json"
)
```

这个入口仅接受原发布提交`0e1cd7f`的实现指纹，核对模型/损失/训练代码保持不变，确认未有训练checkpoint、原始数据与权重未变、全部训练缓存校验通过后，迁移本轮study与进度指纹。原study和进度文件先完整备份，已有NPZ不删除、不覆盖。

它明确禁止新增FM推理。日志应显示156条`REUSED, no inference`，然后整理原验证配对；这时GPU没有负载是正常的。不要重新prepare，也不需要重新执行156条训练缓存生成。

成功标志：`repair_report.json`中`train_records_reused:156`、`cache_completed:true`。`cache.json`应为`completed:true`，同时记录40条唯一验证目标和原配对数。两者数量不同，不代表验证患者变多。

备份：

```text
shared/study.before_pair_cache_fix.json
cache/progress.before_pair_cache_fix.json
```

差异审计：`cache/validation_fm_differences.csv`记录重复目标历史预测的最大电压差、RMSE差及是否超过旧检查阈值。差异原因需要根据该表进一步判断；修复没有假定它只是浮点误差，也没有选择更高分的预测。

如果恢复中断，可重复同一个命令。若提示原版本未知、原数据变化、缓存损坏或训练已经开始，应保留文件与报错，不手工改checksum。

## 3．缓存完成后，再运行原训练命令

GPU0，对照组：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

bash scripts/run_b5_slow_pcc.sh \
  --study "$run/shared/study.json" \
  --cache "$run/cache" \
  --group control \
  --output "$run/control" \
  --gpu 0
```

另一终端GPU2，实验组：

```bash
cd /home/qht/huawei_B5_best_exp_slow_pcc
run=/home/qht/huawei/results/B5-best-exp-SlowPCC/combo_pcc_v1

bash scripts/run_b5_slow_pcc.sh \
  --study "$run/shared/study.json" \
  --cache "$run/cache" \
  --group pcc \
  --output "$run/pcc" \
  --gpu 2
```

第0轮仍须通过既定参考r检查后才更新参数。这次失败在训练初始化之前，因此使用上述普通启动命令，不是`--resume last.pt`。

本地仅完成合成测试；服务器真实缓存的恢复是否成功，以`repair_report.json`和第0轮核对为准。
