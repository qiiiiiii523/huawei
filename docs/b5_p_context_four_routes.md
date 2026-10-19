# P最佳完整模型＋固定context：四组评估

## 1. 本轮固定什么

- 基础完整模型：`/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt`。
- 该文件包含control100 FM＋P慢CNN；不使用同目录的单独慢CNN `best.pt` 作为完整模型。
- body：`/home/qht/huawei/results/B5-best-context/body_verification_v1/o16/O16/best.pt`，O16-seed42-best27，α＝1。
- body历史编码：原共享I尺度归一化、心搏模板截断16、统计截断8。
- d6：原一致性投影＋个人FIR25，不改。
- Heun16/K16、推理噪声seed42、慢分量β＝0.75、120秒拼接、可靠I及目标导联QC、目标记录等权和Task2两设备等权不变。

四组共用一次生成的基础预测：

|group|处理|
|---|---|
|baseline|P完整模型，不加context|
|d6_only|P＋d6 FIR25|
|body_only|P＋固定O16 body修正|
|both|P＋d6＋固定O16 body修正|

这是**冻结旧body修正器，在新P基础模型上的迁移评估**。不训练FM、慢CNN或body，不保证旧body的收益自动保留。旧body训练基础模型的SHA不修改；新入口通过显式 `--allow-base-transfer` 记录两份基础模型SHA，同时严格检查原body权重、编码、验证目标、QC和历史信息一致。其他部署入口原有的基础模型匹配检查保持不变。

先看这四组结果；如果旧O16在P上不再有效，再考虑用P的训练集预测缓存重新训练同一O16方法。当前不要同时重训，以免混淆“直接迁移是否有效”。本轮分数是本地验证集成绩，不能称为主办方隐藏测试集成绩。

## 2. 更新服务器代码

本地工作树：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Context`。

手动上传以下两个新增执行文件，保留目录结构：

```text
本地 scripts/run_b5_p_context.sh
→ /home/qht/huawei_B5_best_context/scripts/run_b5_p_context.sh

本地 baselines/B5_best_context/four_routes.py
→ /home/qht/huawei_B5_best_context/baselines/B5_best_context/four_routes.py
```

本文档可以上传到服务器同工作树的`docs/`，不参与执行。如果通过Git更新，先自行提交推送上述文件，再在服务器对应分支拉取；脚本本身不联网、不切换分支。

## 3. 一条脚本完成生成与四组评估，GPU1

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

sed -i 's/\r$//' scripts/run_b5_p_context.sh
bash -n scripts/run_b5_p_context.sh

bash scripts/run_b5_p_context.sh \
  --repository-root /home/qht/huawei \
  --checkpoint /home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt \
  --run /home/qht/huawei/results/B5-best-context/P_O16_four_v1 \
  --gpu 1
)
```

数据配置默认读取原华为`run_config.yaml`；公开慢CNN实验的`shared/study.json`不是这个数据配置，不能传给`--config`。`stage2_step000390`原评估不用于本轮生成缓存；直接用正式完整`model.pt`重生成，保证最终部署过程与四组输入一致。

脚本结束即完成，不用另跑三个分支，也不用训练。脚本运行时先看到基础生成进度，之后才是四组评估；GPU空闲不等于退出，部分历史特征和FIR在CPU计算。

## 4. 查看结果

```bash
run=/home/qht/huawei/results/B5-best-context/P_O16_four_v1
study=$(cat "$run/study_path.txt")
cat "$study/summary.csv"
cat "$study/four_routes.json"
```

主要看`task1_r`、`machine_r`、`body_r`、`task2_r`、`mean_r`。`mean_r=(task1_r+task2_r)/2`。

- 四组Task1应完全一样，因为当前context只处理Task2。
- `body_only−baseline`与`both−d6_only`的平均r增量应一致。
- `d6_only−baseline`与`both−body_only`的平均r增量应一致。
- 不把此前旧基础模型的context增量直接加在P的新分数上。

## 5. 中断恢复

如果基础生成已经完成，仅后续四组评估中断，用相同设置加`--resume`：

```bash
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_p_context.sh \
  --checkpoint /home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt \
  --run /home/qht/huawei/results/B5-best-context/P_O16_four_v1 \
  --gpu 1 \
  --resume
```

若基础生成本身中断，脚本保留不完整目录并停止，改为一个新`--run`重跑；不会把不完整基础预测用于评分。若修改模型、body、配置或缓存，也必须用新目录。

## 6. 打包报告

```bash
tar -czf /home/qht/B5_P_O16_context_reports.tar.gz \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='*.pt' \
  -C /home/qht/huawei/results/B5-best-context \
  P_O16_four_v1
```
