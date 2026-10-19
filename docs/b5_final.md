# B5-final：固定当前最佳组合

## 1. 这个分支固定什么

从 `B5-best-context` 的 `22a17ca` 创建，统一入口为 `baselines.B5_final`，配置为 `configs/b5_final.json`。
这是已完成实验的最佳组合整理，不增加新网络，也不需要重新训练。

| 部分 | 固定选择 |
|---|---|
| 完整基础模型 | 扩充公开数据训练的 FM-best45＋P最佳慢CNN |
| 快慢组合 | FM快残差＋0.75×预测慢曲线；完整120秒组合 |
| 采样 | Heun16步、K16、推理噪声seed42 |
| Task1 | 基础模型输出，不使用context修正 |
| Task2心电图机 | 原d6一致性投影＋FIR25；个人历史FIR占25% |
| Task2体脂秤 | B去漂移历史＋O16编码＋新FM适配seed43-best3，α＝1 |
| body修改范围 | 仅II；当前I和其他导联不因body修正而变化 |
| 评分 | 可靠输入I与逐目标导联QC；目标记录等权，Task2两设备等权，最终 `(r1+r2)/2` |

这里的body seed43是**追加适配训练的seed**；推理噪声seed仍为42。
适配从旧B43-best15出发，追加10轮中第3轮最佳；不是用原B43-best15直接推理。

## 2. 权重在哪里

服务器原数据/结果根目录仍是 `/home/qht/huawei`。
完整基础模型：

```text
/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt
```

body修正器：

```text
/home/qht/huawei/results/B5-best-context/newFM_body_adapt_v1/adapt/best.pt
```

配置保存两份文件的完整SHA256。入口会拒绝不同权重，即使文件也叫best.pt。
完整基础模型已含FM、慢CNN、归一化尺度与采样设置；不要传单独慢CNN的best.pt。

Git只保存代码、选择配置、操作文档，**没有把服务器.pt权重复制到本机或提交Git**。
两个.pt仍需保留在服务器；运行推理不需要训练缓存、训练集或真实缺失导联。
原实验代码/文档作为历史仍保留；使用本文件的B5_final入口，旧入口默认值未改。

## 3. 已有成绩与选择依据

同一新FM＋P、同一数据/QC、Heun16/K16、β＝0.75：

| 方案 | Task1 r | Task2 r | 最终平均r |
|---|---:|---:|---:|
| 基础模型，不加context | 0.70812168 | 0.70299223 | 0.70555696 |
| 仅d6 | 0.70812168 | 0.70647262 | 0.70729715 |
| d6＋旧B43直接迁移 | 0.70812168 | 0.70687564 | 0.70749866 |
| d6＋body追加适配seed42-best2 | 0.70812168 | 0.70690412 | 0.70751290 |
| **本分支：d6＋body追加适配seed43-best3** | **0.70812168** | **0.70690564** | **0.70751366** |

本分支机器组r＝0.70845323，体脂秤组r＝0.70535804。
两次适配均有小收益，seed43仅略高，不能认为seed43普遍更好。
这些是已有报告成绩；创建分支未在本机重新运行真实ECG评估。

## 4. 首次在服务器建立工作树

先在本机推送 `B5-final`。服务器中保留原 `/home/qht/huawei` 的数据、结果和现有分支，执行：

```bash
cd /home/qht/huawei
git fetch gitee B5-final
git worktree add --detach /home/qht/huawei_B5_final gitee/B5-final
cd /home/qht/huawei_B5_final
conda activate BioFlow
```

该命令用于首次创建；若工作树已存在，在里面先检查工作区再更新：

```bash
cd /home/qht/huawei_B5_final
git status --short
git fetch gitee B5-final
git switch --detach gitee/B5-final
```

有未保存改动时，先自行保存改动，不使用reset/clean覆盖。

## 5. 检查选中的权重

```bash
cd /home/qht/huawei_B5_final
python -m baselines.B5_final.profile --repository-root /home/qht/huawei
```

输出 `ready: True` 表示两份文件及SHA256正确；这是文件检查，不是模型精度或前向检查。
目录搬迁时用 `--repository-root` 改结果根目录，或者用 `--checkpoint`、`--body-checkpoint`
指定两份同内容文件的新位置；不会自动寻找“最新best”或换模型。

## 6. 一条命令算四组r，GPU1

默认复用已完成的新FM基础预测缓存，重新应用context并算四组分数：

1. baseline：无context。
2. d6_only：仅机器d6。
3. body_only：仅选定body。
4. both：选定完整组合。

```bash
cd /home/qht/huawei_B5_final
bash scripts/run_b5_final.sh --gpu 1
```

默认结果生成到 `/home/qht/huawei/results/B5-final/evaluation_时间_PID/`。
结束会打印真实目录。`study/summary.csv` 是四组r，`study/final.json` 是选定权重、设置、
数据指纹、四组结果和历史最佳成绩复现检查。
同一记录在各组使用相同基础预测；脚本不训练，不联网，不切换分支。
相同目标/QC下要求完整组合与历史报告各r相差不超过1e-7。
若检查失败会保留报告并报错，不把结果直接当作历史最佳已复现。

如需从权重**重新生成基础预测**再做四组评估：

```bash
cd /home/qht/huawei_B5_final
bash scripts/run_b5_final.sh --gpu 1 --regenerate-base
```

每次使用新结果目录。失败后也保留原结果，重新执行生成新目录，不覆盖。
也可通过 `--run /home/qht/huawei/results/B5-final/自定义新目录` 指定。

若后续换了验证人群或QC规则，不能直接要求复现旧分数。需准备该数据的纯基础预测，再显式运行
`python -m baselines.B5_final.validate --config ... --baseline-evaluation ... --output ...`，
不加 `--verify-reference`；报告仍会注明人群是否相同。

## 7. 单条真实输入预测

输入为同步I、人口学信息，以及该设备的**原始历史**；历史不与当前120秒强制对齐。
当前I是500Hz、μV、[1,60000]或[60000]。历史六导联顺序为I、II、III、aVR、aVL、aVF，
形状[6,T]，另提供同形状布尔mask，包含导联QC与缺失片段。
不得先将body历史做B滤波再输入；权重保存的预处理会从原始历史执行一次滤波。

体脂秤示例（把input路径替换成实际文件，output必须是新目录）：

```bash
cd /home/qht/huawei_B5_final
CUDA_VISIBLE_DEVICES=1 python -m baselines.B5_final.predict \
  --anchor-npy /home/qht/input/current_I_uV.npy \
  --metadata-json /home/qht/input/demographics.json \
  --record-key 实际稳定记录ID \
  --task-id task2 \
  --input-type body_scale_d6 \
  --history-npy /home/qht/input/raw_body_history_uV.npy \
  --history-mask-npy /home/qht/input/body_history_valid_mask.npy \
  --history-record-id 实际历史记录ID \
  --output /home/qht/huawei/results/B5-final/实际记录预测新目录 \
  --device cuda
```

机器组改为 `--input-type ecg_machine_d6` 并提供机器原始历史；Task1使用
`--task-id task1 --input-type watch_ecg`，不传历史。人口学字段与B5既有接口一致。
输出 `prediction_uV.npy` 为[12,60000]，`inference.json`记录权重、推理参数和实际context使用/回退状态。
未提供可用历史时沿用已评估实现的回退：机器仅投影，body使用学习到的通用II修正；
报告会注明，不能把它解释为利用了个人历史。

## 8. 打包评估报告

把run设置为脚本结束打印的真实目录：

```bash
run=/home/qht/huawei/results/B5-final/填写本次实际目录
tar -czf /home/qht/B5_final_reports.tar.gz \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
  -C "$run" preflight.json preflight.log study evaluate.log
```

这份报告不包含权重或大数组。当前分支不提供重新训练入口；以后优化应再建立实验分支。
