# 新完整模型＋当前最佳context：三组评估

本轮只评估，不重训。三组共用同一批完整120秒预测、Heun16/K16、推理seed42、慢分量β＝0.75、可靠I和目标导联QC、目标记录等权及Task2两设备等权规则。

|组件|本轮固定路径／设置|
|---|---|
|新完整模型|`/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt`|
|body|`/home/qht/huawei/results/B5-best-context/P_body_AB_seed43_v1/B/best.pt`，训练seed43、best15|
|body训练特征缓存|`/home/qht/huawei/results/B5-best-context/P_body_AB_v1/prepared/B_cache`|
|body编码|B整记录0.2Hz去漂移＋O16，α＝1|
|d6|原一致性投影＋FIR25|

这是冻结B43 body迁移到新FM的实验。旧body的原训练模型SHA和缓存不修改，新评估单独记录新基础模型SHA。基础权重变了，之前0.707059的组合分数不能直接沿用。

## 1. 推送后更新服务器

本地工作树：`C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Context`，分支`B5-best-context`。

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_best_context
git fetch gitee B5-best-context
git merge --ff-only FETCH_HEAD
git log -1 --oneline
sed -i 's/\r$//' scripts/run_b5_selected_context.sh
bash -n scripts/run_b5_selected_context.sh
)
```

## 2. GPU1：一次生成＋三组评分

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_selected_context.sh \
  --checkpoint /home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --run /home/qht/huawei/results/B5-best-context/FM_P_best45_B43_v1 \
  --gpu 1
)
```

不使用旧的`run_b5_p_context.sh`或`four_routes.py`，它们仍固定旧O16-best27，不会被这次新入口悄悄改变。

默认重新生成新完整模型的基础预测，只生成一次。如果已经有该**同一完整模型**的正式reliable-I验证目录，且包含两任务`prediction_uV.npy/target_uV.npy/target_quality_mask.npy/window_metadata.csv`，可以在首次运行时额外传`--baseline-evaluation /实际新评估目录`复用。不要传旧P的评估目录或只有分数、没有波形数组的中间训练评估。入口会核对权重、目标记录和QC指纹，防止用错缓存。

## 3. 查看结果

```bash
run=/home/qht/huawei/results/B5-best-context/FM_P_best45_B43_v1
study=$(cat "$run/study_path.txt")
cat "$study/summary.csv"
cat "$study/selected_context.json"
```

|group|方案|
|---|---|
|baseline|新完整模型，不加context|
|d6_only|新完整模型＋d6|
|both|新完整模型＋d6＋冻结B43 body|

三组Task1应完全一致，`d6_only`和`both`的机器组r应相同。`both−d6_only`是body在新基础模型上的额外贡献。如果body降低平均r，不自动部署它；先保留d6方案，再考虑重新适配body。三组评估不会自动更换项目默认权重。

## 4. 中断恢复

基础生成完成、后处理中断时，原命令末尾增加`--resume`即可。基础生成本身中断时，不使用不完整数组，保留原目录并选择新`--run`重跑。修改基础权重、数据、body或缓存时也要新建输出目录。

```bash
cd /home/qht/huawei_B5_best_context

bash scripts/run_b5_selected_context.sh \
  --checkpoint /home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --run /home/qht/huawei/results/B5-best-context/FM_P_best45_B43_v1 \
  --gpu 1 \
  --resume
```

## 5. 打包

```bash
tar -czf /home/qht/B5_newFM_B43_context_reports.tar.gz \
  --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  -C /home/qht/huawei/results/B5-best-context \
  FM_P_best45_B43_v1
```

生产推理传原始完整可见body历史；由B43权重保存的预处理协议滤波一次再提取特征，不传预滤数据以免重复滤波。本轮不会读取当前隐藏目标作为模型输入，不对预测做目标均值校正或时间对齐。
