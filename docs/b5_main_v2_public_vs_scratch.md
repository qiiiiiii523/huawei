# 新版 B5：公开预训练作用的两个实验

本次仅比较初始化方式：E1华为数据从头训练，E2公开预训练EMA初始化后在相同华为数据上微调。
两组均使用同步I＋人口学的原B5-U，保留原FM损失、100轮、LR=5e-5、batch=8、seed42、每5轮验证K1、最终Heun16/K16。
不增加T/F/context，不修改损失权重，也不把周期验证改成K16或每轮验证。

这比较的是“额外公开预训练在相同华为训练预算下是否有帮助”。E2总计算量包含已经完成的公开预训练，不能声称两组总计算预算相同。
从头训练组使用相同LR和预算是控制变量的起点，不证明已找到从头训练的最优超参数；一个seed的结果也不代表所有运行。

## 两组

| 项目 | E1：华为从头训练 | E2：公开初始化＋华为微调 |
|---|---|---|
| 配置 | b5_main_v2_scratch_meta.yaml | b5_main_v2_public_finetune_meta.yaml |
| stage | local | finetune |
| 初始化 | 固定seed的随机权重，不传init-checkpoint | E2_public_meta/best.pt的EMA |
| 华为训练 | 同一strict去重1872窗/156条d12 | 同左 |
| 人口学 | data/userinfobean.csv，已更新的文件 | 同左 |
| scale | 同一冻结preprocessing_scales.npz | 同左，必须与公开权重sha匹配 |
| 选模 | 每5轮K1验证，总分最大；另保存相关性/加分最佳 | 同左 |
| 最终评估 | K16，完整新版验证集 | 同左 |
| 输出 | results/B5/main_v2_score60_scratch_meta | results/B5/main_v2_score60_public_finetune_meta |

两个配置只在stage、初始化要求、实验ID和输出目录上不同，旧b5_base/b5_finetune配置的超参数不变。
E1禁止传init-checkpoint，防止把“华为从头训练”误跑成带预训练。E2要求public来源的checkpoint，不能误用旧华为微调best。
两组均可resume各自采用新选模协议生成的last.pt，不能跨stage、换人口学、换数据或换采样后resume。旧选模协议last仅能用于新运行初始化，不能精确续训。

## 服务器先检查

用户提交/推送本地B5后，服务器更新代码并保留自身路径修改：

```bash
cd /home/qht/huawei
git fetch origin
git switch B5
git pull --ff-only origin B5
python -m unittest discover -s baselines/B5/tests -v

python -m baselines.B5.prepare \
  --config configs/experiments/b5_main_v2_scratch_meta.yaml \
  --report results/B5/main_v2_pair_scratch_preflight.json

python -m baselines.B5.prepare \
  --config configs/experiments/b5_main_v2_public_finetune_meta.yaml \
  --report results/B5/main_v2_pair_public_preflight.json
```

检查ready=true，人口学实际路径为huawei/data/userinfobean.csv，行数185，两个预检人口学sha相同且为新文件，训练1872窗、验证324/780窗。
预检新增各数据集按唯一患者统计的字段覆盖率；185行不等于185位信息完整患者。正确CSV仍缺少部分患者/字段，用mask，不虚构补值。
预检不拟合新scale、不训练。继续复用公开checkpoint对应的既有冻结scale，不要为E1单独换另一套scale。

## 单卡分别训练

可先E1再E2，在同一个终端顺序执行；每组只使用一张GPU。

```bash
set -o pipefail
mkdir -p results/B5/main_v2_score60_scratch_meta results/B5/main_v2_score60_public_finetune_meta

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_main_v2_scratch_meta.yaml \
  --device cuda --execute-training \
  2>&1 | tee results/B5/main_v2_score60_scratch_meta/train.log

CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_main_v2_public_finetune_meta.yaml \
  --init-checkpoint results/B5/E2_public_meta/best.pt \
  --device cuda --execute-training \
  2>&1 | tee results/B5/main_v2_score60_public_finetune_meta/train.log
```

并行时分别在两个终端运行一组、指定不同空闲卡。直接把两条前台命令粘贴到同一个终端仍是顺序执行。
输出目录不得已有best/best_r/best_bonus/last；重跑换新目录，不能覆盖此前人口学错误的实验。
不要resume人口学错误的旧run；本次两组都从各自定义的起点开始。

## 训练后评估

两组采用同样的四份权重保存规则，best.pt按50分主分＋10分加分的总分选择，最终统一K16。四份候选全部评估，检查K1选模与K16实际使用是否出现排名变化；不要把额外评估误称为训练用了K16。具体规则见[评分与保存说明](b5_score60_checkpoint_policy.md)。

```bash
for arm in scratch public_finetune; do
  if [ "$arm" = scratch ]; then
    config=configs/experiments/b5_main_v2_scratch_meta.yaml
  else
    config=configs/experiments/b5_main_v2_public_finetune_meta.yaml
  fi
  for checkpoint in best best_r best_bonus last; do
    CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
      --config "$config" \
      --checkpoint "results/B5/main_v2_score60_${arm}_meta/${checkpoint}.pt" \
      --device cuda \
      --output-dir "results/B5/main_v2_score60_${arm}_meta/evaluation_${checkpoint}"
  done
done
```

比较同一新版验证集的Task1 r、Task2两设备r及等权r、两设备胸导联RMSE和综合分。不能直接对比旧数据0.66949等历史分数。
initialization.json中E1应type=random、source_checkpoint=null；E2应type=pretrained_ema、source_stage=public。
两组的人口学、数据manifest和scale sha必须相同。
K1选模与K16评价的限制是有意保留的旧协议，应在报告中说明；本次不修改它，后续另设实验才改变。

## 打包报告

```bash
tar --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  -czf B5_main_v2_pair_reports.tar.gz \
  results/B5/main_v2_score60_scratch_meta \
  results/B5/main_v2_score60_public_finetune_meta \
  results/B5/main_v2_pair_scratch_preflight.json \
  results/B5/main_v2_pair_public_preflight.json \
  configs/experiments/b5_main_v2_scratch_meta.yaml \
  configs/experiments/b5_main_v2_public_finetune_meta.yaml \
  configs/experiments/b5_base.yaml configs/b5_inference.yaml
```

本机不启动训练；代码具备两组运行能力不等于实验已经得到成绩。当前版本增加了正确60分评分与四份权重保存，两组规则完全相同；公共预训练不用重跑。此前已完成的Task1选模实验保留在旧目录，报告可以不训练、不生成波形而直接重算总分。
