# B5-scratch：公开FM初始化，华为微调第1轮加入冻结慢条件

名字按用户要求叫B5-scratch；实验不是随机初始化从零训练。它从已完成的公开预训练FM开始，新增零初始化慢条件适配器，FM主干和适配器从华为微调第1轮共同训练。S50/S50-F CNN冻结；网络接入位置、原FM损失、人口学、尺度、学习率不另改。默认100轮、每5轮Heun16/K16验证，以两个任务平均r选best_r.pt，另存last.pt和历史。RMSE不算分。

本分支从B5-best创建，独立模块为baselines.B5_scratch。正在服务器运行的B5-S50三组续训继续在原worktree运行，不切换它，不覆盖结果。新分支还保留B5-best稳定推理入口，但当前实验checkpoint是接入模型，不能直接冒充原B5-U组合checkpoint。

## 1. 独立更新代码

本机提交后手动推送：

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Scratch'
git push --set-upstream origin B5-scratch
```

服务器使用已配置且已同步此分支的gitee远程：

```bash
cd /home/qht/huawei
git fetch gitee B5-scratch
git worktree add --detach /home/qht/huawei_B5_scratch FETCH_HEAD
cd /home/qht/huawei_B5_scratch
conda activate BioFlow
bash -n scripts/run_b5_scratch.sh
```

已存在时，在新worktree中fetch并switch --detach FETCH_HEAD，不重复创建。不要在正在续训的/home/qht/huawei_fm_S50中切分支。

## 2. 准备新的独立缓存＋公开模型第0轮参考

**现在可以准备代码；占用GPU的prepare和训练等空闲卡再运行。新建worktree不会停止旧进程，但两个进程使用同一GPU会争显存。**

需要原华为best.pt、原prepared、S50和S50-F源文件，以及已有公开预训练checkpoint。无需重跑公开预训练。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_scratch
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
root=/home/qht/huawei/results/B5-scratch/early_public_v1
mkdir -p "$root"

python -u -m baselines.B5_scratch.prepare \
  --reference-checkpoint /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt \
  --slow-checkpoint /home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints/best_score.pt \
  --slow-f-checkpoint /home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/best_score.pt \
  --prepared /home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared \
  --epochs 100 --output "$root/shared" --device cuda \
  2>&1 | tee "$root/prepare.log"

python -m baselines.B5_scratch.check \
  --shared "$root/shared" --report "$root/preflight.json" \
  2>&1 | tee "$root/preflight.log"
)
```

公开权重默认从reference-checkpoint内initialization.source_checkpoint自动定位，并核验原SHA。该路径不在时，prepare会明确停止：在命令里增加 `--public-checkpoint /实际公开训练结果目录/best.pt`。不会扫描目录随便挑权重；也不能把华为best.pt传进去。

`--reference-checkpoint`只用于核对已有数据和尺度，不用于初始化。shared.json必须记录public_source_stage=public、initialization_mode=public_pretrained_early_slow；fm_checkpoint/fm_sha256指向公开权重。程序额外用公开模型生成真实第0轮Heun16/K16参考，三组起始评分必须相同；不能拿旧华为best的r当新第0轮成绩。

准备没有优化器更新。缓存位于results/B5-scratch，不复用旧fm_slow_qc_three_v2/shared，不写原S50缓存。成功后不重复prepare。失败保留目录，用新root重做并让后面全部传同一root。

## 3. 卡空闲后，先跑control和slow

终端A，空闲GPU0：

```bash
cd /home/qht/huawei_B5_scratch
conda activate BioFlow
bash scripts/run_b5_scratch.sh --group control --gpu 0
```

终端B，空闲GPU1：

```bash
cd /home/qht/huawei_B5_scratch
conda activate BioFlow
bash scripts/run_b5_scratch.sh --group slow --gpu 1
```

两组同公开EMA初始化、相同新优化器、100轮和shuffle/noise seed；slow仅增加冻结S50条件＋可训练适配器。S50-F接入作为可选第三组，空闲GPU2执行 `bash scripts/run_b5_scratch.sh --group slow_f --gpu 2`，不是必须先跑。

脚本训练后只评估best_r和last。评估同时计算FM完整输出，以及“该FM快残差＋同一个冻结S50-F慢预测器”的完整输出，均按相同QC与平均r比较。组合不使用目标慢标签；best_r按FM直接输出选，不声称它也是所有epoch组合输出的最优。

进程中断后原命令加 --resume，仅恢复B5-scratch该组last.pt；不能恢复B5-S50或旧B5的last.pt。若评估中断，先改名保存对应不完整输出目录，再单独执行evaluate或续跑脚本，不覆盖。脚本不是后台服务；需要断开终端时可自行使用tmux。

## 4. 同预算完成后汇总

```bash
cd /home/qht/huawei_B5_scratch
root=/home/qht/huawei/results/B5-scratch/early_public_v1
python -m baselines.B5_scratch.compare \
  --shared "$root/shared" \
  --control "$root/control" --slow "$root/slow" \
  --output "$root/comparison"
```

如果跑了第三组，加 `--slow-f "$root/slow_f"`。查看comparison/report.md和scores.csv：先比较slow相对同预算control的FM平均r，再比较hybrid_slow_f组合是否超过旧稳定组合0.69262935。后者只是固定验证集参照；新测试集结果不能预先保证。

打包报告：

```bash
cd /home/qht/huawei
find results/B5-scratch/early_public_v1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' -o -name '*.png' \) \
  -print0 | tar --null -T - -czf B5_scratch_reports.tar.gz
```

## 5. 隔离范围

原main、B5、B5-best、B5-S50工作树与其结果都不修改。新训练仅在明确传 --execute-training 时启动。本地功能检查使用CPU合成数据，没有训练和优化器更新；真实公开权重、数据和GPU表现需在服务器预检及运行后确认。

本地66项CPU功能及回归检查通过，1项需要真实缓存的检查跳过。新实验检查覆盖公开/华为初始化分离、尺度及架构不匹配拦截、第0轮公开模型参考、零初始化ODE等价和投影梯度、旧S50检查点/缓存拒绝、最终评估复用、相同S50-F的输出组合及同预算两组汇总。文件流程中的采样和完成轮次使用合成替身，不代表真实训练或性能结果。
