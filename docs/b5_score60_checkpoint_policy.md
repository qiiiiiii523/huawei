# B5：50分主分＋10分加分与四份checkpoint

修正公共评分，而非在B5重复实现另一套公式：

```text
main_correlation = 0.5*r1 + 0.5*r2
main_score = 50*main_correlation
bonus = 0.5*b(machine_RMSE) + 0.5*b(body_RMSE)
b(x) = 10 (x<=70 μV)，否则700/x
competition_total_score = main_score + bonus
```

相关性满分时主分50分，两设备加分满分时bonus10分，此模型评分最高60分。旧1.x总分漏乘50，不能沿用；r、RMSE和已训练权重无需因此重算或重训。

## 权重规则

| 文件 | 刷新规则 | 用途 |
|---|---|---|
| best.pt | competition_total_score最高 | 总分候选 |
| best_r.pt | main_correlation最高 | 两任务相关性均值候选 |
| best_bonus.pt | task2_rmse_bonus_score最高 | 两设备等权加分候选 |
| last.pt | 每个训练epoch保存 | 优化器/EMA/RNG与三项最佳状态的续训 |

三个best独立刷新，同分保留先前权重；同一轮可能同时刷新三个文件。首次验证前只有last，不生成虚构的best。
保存EMA和原训练模型，以及完整训练状态；评估仍默认用EMA。新checkpoint为format_version=3，记录评分协议、选模协议、各指标最佳值及其轮次，best_score现在明确指总分。
每次验证的完整r1/r2、设备RMSE、加分、主分、总分继续写入history.jsonl，并新增validation_metrics.csv。选择文件不删除指标曲线。

## 采样与控制变量

按用户确认，周期验证和最终validate/predict均为Heun16/K16、seed42；每5轮验证。网络、条件、FM训练目标、损失、学习率、100轮预算不变。
best.pt按实际验证过的轮次的K16总分最高保存，不保证未验证轮次没有更好的权重。K16仅影响生成与验证，并非每个训练样本重复训练16次。
验证速度场调用数约为K1的16倍；生成逐样本累计取平均，条件编码复用，不把16份生成同时堆入显存。

旧K1选模实验仍保留在旧目录。新运行使用score60_k16目录：

```text
results/B5/main_v2_score60_k16_scratch_meta/
results/B5/main_v2_score60_k16_public_finetune_meta/
```

使用现有两个实验配置及train_huawei入口，命令结构不变。E1不传初始化，E2仍用公开best初始化。不要覆盖已完成实验。

旧checkpoint可以推理或初始化新运行，但不能直接resume：旧best_score是相关性或旧单位总分，没有完整三项ledger。
新规则精确resume只使用本次K16实验的last.pt，采样/数据/尺度/阶段等也须一致。若要从旧华为权重开启新一段微调，使用不限定public来源的b5_finetune_meta.yaml并另设输出目录；配对E2配置仍限定公开权重，以保护消融含义。

## 重算已有报告：不训练、不生成波形

在包含修正公共公式的代码版本中执行（示例为已完成公开微调的last）：

```bash
cd /home/qht/huawei
python scripts/summarize_competition_score.py \
  --task1-overall results/B5/main_v2_public_finetune_meta/evaluation_last/task1/overall_metrics.csv \
  --task2-overall results/B5/main_v2_public_finetune_meta/evaluation_last/task2/overall_metrics.csv \
  --output-dir results/B5/main_v2_public_finetune_meta/evaluation_last/score60
```

已有best/last的r与设备RMSE足以重算正确总分，不需要重新跑K16生成。之前没有保存的中间轮权重无法从日志恢复，因此不能补齐历史全局最优的四个模型。

本次报告重算：华为从头第100轮31.800319；公开微调第5轮32.860461；公开微调第100轮32.941415。后者仅高0.080954分，不能继续沿用旧单位下的差值判断。

## 评估四份候选

```bash
for checkpoint in best best_r best_bonus last; do
  CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
    --config configs/experiments/b5_main_v2_public_finetune_meta.yaml \
    --checkpoint "results/B5/main_v2_score60_k16_public_finetune_meta/${checkpoint}.pt" \
    --device cuda \
    --output-dir "results/B5/main_v2_score60_k16_public_finetune_meta/evaluation_${checkpoint}"
done
```

目录须为空，最终采样仍K16。若某文件不存在，说明尚未成功执行首次验证，不能跳过错误并当作已训练完成。

## 分支工作目录

main公共评分修正已提交并推送为f607fb1（修正评分V3_10.11），临时main工作树已删除。B5已包含同样公共文件，并独立实现四份权重保存。
本次K16周期验证修改在本机B5工作目录，未提交、未推送；无需再次修改main。
