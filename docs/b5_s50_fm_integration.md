# B5-S50：冻结慢预测器，作为条件接入 Flow Matching

本分支从最新 `B5`（7c38682）创建。它实现一轮配对微调实验，不是已经验证有效的新模型。本地不启动真实训练。

## 1. 这轮比较什么

|设置|control 对照|slow 实验|
|---|---|---|
|FM初始化|指定的原 B5 `best.pt` 中的EMA|完全相同|
|输入|10秒同步I、人口学、噪声状态、生成时间t|另外加入冻结S50预测的11导联慢曲线|
|新增网络|无|11→32→256的小卷积条件编码器，9536个参数|
|接入位置|原U-Net|瓶颈相加；最后投影零初始化|
|训练|原FM及原辅助损失，新优化器，新增20轮|相同；S50不更新|
|周期/最终生成|Heun16/K16，固定seed42，拼接120秒|相同|
|选模|总分best、相关性best、bonus best、last|相同；共同初始化第0轮也可入选|

学习率、batch、梯度累计、EMA、训练质量mask、人口学、尺度都继承原FM检查点，不自行换成慢网络训练配置。每5轮验证。K16是生成验证/推理次数，训练的FM噪声插值目标没有改成16次。

结构：`完整120秒I＋人口学 → 冻结S50 → 120秒预测慢曲线 → 裁当前10秒 → 编码并加入FM瓶颈 → 原解码器 → 11导联速度场`。I输出头保持原路径。

S50在50Hz下读取50秒I，中心输出10秒。首尾缺少邻居时重复端点；拼成120秒后，按全记录采样中心插值回500Hz。没有逐窗去均值，没有额外电压放大，没有真实慢标签/真实均值作为输入。训练时预计算并缓存，采样时不重复运行16次S50。部署检查点内嵌冻结S50，遇到新记录仍从I自己预测条件。

**评分口径：本轮明确用此前S50报告的完整集合v2** `correlation50_bonus10_target_record_v2`，总分为50×(r1+r2)/2，加上两设备等权RMSE加分，最高60分。最新B5默认QC过滤评分不用于这轮配对；本分支不改它的默认设置。训练仍使用原质量mask；诊断同时报告正常/全部导联。不能将本轮分数与QC过滤分数直接混比。

## 2. 在服务器建立独立代码目录

本地提交后需要把 `B5-S50` 推送到服务器可访问的远程。服务器用实际配置的远程，例如gitee：

```bash
cd /home/qht/huawei
git fetch gitee B5-S50
git worktree add -b B5-S50 /home/qht/huawei_fm_S50 gitee/B5-S50
cd /home/qht/huawei_fm_S50
conda activate BioFlow
sed -i 's/\r$//' scripts/run_b5_s50_pair.sh
bash -n scripts/run_b5_s50_pair.sh
```

如果服务器已经有本地 `B5-S50` 分支，把 worktree 命令改为 `git worktree add /home/qht/huawei_fm_S50 B5-S50`；不要强制覆盖已有worktree。若远程叫origin，用对应名称替换gitee。脚本内部不执行fetch，不要求访问GitHub。

## 3. 一次性准备，两组共用

原B5、原S50、prepared及原评估文件都只读，不改旧实验。下面FM路径是你纠正后的 **best.pt**。

```bash
cd /home/qht/huawei_fm_S50
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
root=/home/qht/huawei/results/B5-S50/fm_condition_v1
mkdir -p "$root"
set -o pipefail

python -u -m baselines.B5_s50.prepare \
  --fm-checkpoint /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt \
  --slow-checkpoint /home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints/best_score.pt \
  --prepared /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared \
  --output "$root/shared" \
  --device cuda \
  2>&1 | tee "$root/prepare.log"

python -u -m baselines.B5_s50.check \
  --shared "$root/shared" \
  --report "$root/preflight.json" \
  2>&1 | tee "$root/preflight.log"
```

预期 `completed:true`，随后 `ready:true`、两项difference=0、投影梯度>0、optimizer_updates=0。准备阶段先核验数据/人口学/检查点/评估文件的SHA，CPU读盘可能耗时；之后GPU只做冻结S50推理。不是训练，不要因为暂时没占GPU就重复启动。已经成功后不再跑prepare，现有shared不会被覆盖。

准备会重新用本轮v2规则给原评估预测算分；每组正式微调前还会执行完整K16第0轮验证并核对。若不一致就停止，先检查旧文件/噪声/输入，不跳过检查继续训练。新组的运行配置写在 `control/slow/checkpoints/resolved_config.json`。

## 4. 两个终端分别运行，自动训练后评估四份权重

终端A，GPU0：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh --group control --gpu 0
```

终端B，GPU2：

```bash
cd /home/qht/huawei_fm_S50
conda activate BioFlow
bash scripts/run_b5_s50_pair.sh --group slow --gpu 2
```

每组均为单卡。两个终端可并行；同一个终端顺序粘贴会先跑完第一组。K16周期验证较慢，目录中会先生成 `validation/epoch_0000`。共有20个**新增**epoch，不是把原epoch重置当作从头训练。不要再使用原 `baselines.B5.train_huawei` 入口训练带慢条件的模型。

异常中断后，在同组命令末尾加 `--resume`，从该组 `checkpoints/last.pt` 精确续跑；不能用 `--resume` 加载原B5/S50的权重。训练验证中断目录会自动保留并重跑；若最终 `evaluations/best` 等目录只生成一半，先把那个目录改名为 `best_interrupted`，再续跑。完整评估会保留，同一组相同EMA的候选复用预测，避免重复K16计算。脚本不是后台服务，断连可能停止；可放在自己已有tmux会话里。

## 5. 两组结束后比较

```bash
cd /home/qht/huawei_fm_S50
root=/home/qht/huawei/results/B5-S50/fm_condition_v1
python -m baselines.B5_s50.compare \
  --shared "$root/shared" \
  --control "$root/control" \
  --slow "$root/slow" \
  --output "$root/comparison"
```

看 `comparison/report.md` 与 `scores.csv`：实验组相对同预算对照有无收益、是否超过第0轮，以及r/RMSE的取舍。检查各组 `evaluations/best/wave_summary.csv` 和 `plots`，尤其独有患者、V4、波峰与10秒接缝。冻结S50的历史输出替换组合分若有匹配报告，会作为单独参考列出；它不是本次接入FM的训练成绩。

若有收益，再换训练seed复核。无收益时先分析模型是否利用了条件、慢/快分量变化及优化过程，再考虑融合位置或解冻；不要直接认定慢信息没用，也不马上叠加F/Transformer。当前是旧FM权重上新增条件微调；从头训练可能得到不同结果，但成本更大，属于下一轮独立比较。

打包轻量报告（不含权重、预测数组、原始数据）：

```bash
cd /home/qht/huawei
find results/B5-S50/fm_condition_v1 -type f \( \
  -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o \
  -name '*.md' -o -name '*.log' -o -name '*.png' \) \
  -print0 | tar --null -T - -czf B5_S50_FM_reports.tar.gz
```

## 6. 新记录推理，不需要真实目标或prepared

`new_I_uV.npy` 为500Hz同步I，形状 `[60000]` 或 `[1,60000]`，原始μV；人口学JSON例子为 `{"age":45,"sex":"male","height":170,"weight":65}`，字段允许缺失。输入必须覆盖完整120秒；这轮不是实时因果预测，使用了前后I。

```bash
cd /home/qht/huawei_fm_S50
CUDA_VISIBLE_DEVICES=2 python -m baselines.B5_s50.predict \
  --checkpoint /home/qht/huawei/results/B5-S50/fm_condition_v1/slow/checkpoints/best.pt \
  --anchor-npy /path/to/new_I_uV.npy \
  --metadata-json /path/to/new_metadata.json \
  --record-key new_record_001 \
  --output /home/qht/huawei/results/B5-S50/new_record_001 \
  --device cuda
```

输出 `prediction_uV.npy` 为 `[12,60000]`，默认Heun16/K16。`record-key`固定以复现噪声。训练标签仅用于监督；评估标签仅用于算分、画图。以上推理入口不会读取标签或训练目录。

## 7. 本地已验证的范围

8项新增合成数据功能检查通过；原B5网络、main-v2接口、评分选模与采样的35项回归检查通过，1项按原测试条件跳过。正式64/128/256通道、5000点窗口的速度场零初始化最大差值为0，新增9536个参数。合成数据检查覆盖S50窗口/全局插值、无标签条件、投影梯度、EMA验证不干扰训练RNG、初始化四份候选、续跑恢复、评估复用和仅I/人口学的新记录推理。

没有进行真实ECG训练，没有执行优化器更新；文件流程检查使用合成数据和采样替身，真实ODE等价性另用短合成信号检查。尚未验证服务器真实检查点、数据路径、GPU显存和最终效果，以上第3步预检及第0轮验证用于在服务器核实这些条件。
