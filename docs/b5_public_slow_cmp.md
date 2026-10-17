# 慢CNN公开数据扩充：服务器操作

分支：`B5-best-exp-PublicSlow`，从最新B5-best的`ec12008`创建。

只更新慢CNN，FM固定control100，网络结构、β=0.75、Heun16/K16不改。当前稳定B5-best和旧SlowPCC实验不修改。

## 1. 本轮跑什么

| 项目 | C | M | P |
|---|---|---|---|
| 初始化 | 同一份当前慢PCC20 | 同左 | 同左 |
| 阶段一 | 仅华为 | 每次更新1条华为＋1条PTB | 仅PTB |
| 阶段一损失 | 慢Huber＋0.1×慢PCC＋原F | 相同 | 相同 |
| 阶段二 | 华为，恢复额外0.1×最终组合PCC | 相同 | 相同 |
| 每阶段预算 | 1560次参数更新，记录batch=2 | 相同 | 相同 |
| 阶段交接 | 固定阶段一最后一步，不挑不同公开best | 相同 | 相同 |
| 最终选模 | 初始权重及阶段二候选中，完整120秒平均r最高 | 相同 | 相同 |

阶段二新建AdamW，lr=1e-4、weight_decay=0.01；阶段一也使用同样的优化器设置。每390次更新验证一次。两阶段可以重复使用同一华为训练集，训练患者与验证患者必须隔离。

采用最新可靠I输入＋目标导联QC、目标记录等权、Task2两设备等权评分。旧0.698216来自历史评分范围，不能直接作为新协议硬门槛；三组训练前均重新记录同一初始化分数。

## 2. 本地推送、服务器创建工作树

本地在新工作树推送；本实现不自动推送：

```powershell
cd 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-PublicSlow'
git push -u origin B5-best-exp-PublicSlow
```

当前本地配置的是origin（GitHub），服务器使用gitee。推送后应按你现有方式将该分支同步到Gitee，再执行服务器fetch；本代码不自动同步远程、不替你推送。

服务器首次创建（`/home/qht/huawei_B5_public_slow`尚不存在时）：

```bash
conda activate BioFlow
cd /home/qht/huawei

git fetch gitee B5-best-exp-PublicSlow
git worktree add --detach /home/qht/huawei_B5_public_slow FETCH_HEAD

cd /home/qht/huawei_B5_public_slow
git log -1 --oneline
```

以后只更新这个工作树，先确认没有运行中的实验：

```bash
cd /home/qht/huawei_B5_public_slow
git status --short
git fetch gitee B5-best-exp-PublicSlow
git merge --ff-only FETCH_HEAD
```

study会绑定代码摘要。已开始的实验请在创建study时的提交上完成；升级代码后需要新结果目录和新study，不覆盖旧研究。

## 3. 数据和已有权重放哪里

| 项目 | 路径 |
|---|---|
| PTB Diagnostic原始数据 | `/home/qht/huawei/data/PTB-Diagnostic/patient*/` |
| 当前双模型起点包 | `/home/qht/huawei/results/B5-best-cross/seed42_20261017_094115_417122/control100__pcc20.pt` |
| 已准备的control100缓存 | `/home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache` |
| 本次结果根目录 | `/home/qht/huawei/results/B5-slow-public/CMP_v1` |

先等下载完成，再准备PTB缓存：

```bash
cd /home/qht/huawei_B5_public_slow
python -c 'import torch, numpy, scipy, wfdb; print("dependencies ready")'

find /home/qht/huawei/data/PTB-Diagnostic -maxdepth 2 -name '*.hea' | wc -l
```

如果下载器产生了额外的`physionet.org/files/ptbdb/1.0.0`嵌套，后面`--root`改为实际直接包含`patient001`等目录的那层。不要移动或覆盖原PTB-XL。

本轮不重新训练FM、不修改现有权重。control100缓存只有准备完成并有`cache.json`后才能使用；仅有部分npz不算完成。

## 4. 一次性处理PTB（CPU，不训练）

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_public_slow

run=/home/qht/huawei/results/B5-slow-public/CMP_v1
bundle=/home/qht/huawei/results/B5-best-cross/seed42_20261017_094115_417122/control100__pcc20.pt
mkdir -p "$run"

python -u -m baselines.B5_public_slow.ptb \
  --root /home/qht/huawei/data/PTB-Diagnostic \
  --bundle "$bundle" \
  --output "$run/public" \
  2>&1 | tee "$run/public_prepare.log"

cat "$run/public/audit.json"
)
```

处理规则：

- 按PTB头文件gain、baseline和单位转换到μV；只读取标准12导联。
- 完整记录抗混叠重采样1000→500Hz；不重复缩放、不整体去均值。
- 完整可用记录先做501点居中移动平均，之后再取真实连续片段和降至50Hz。
- 公开记录使用起始连续50～120秒，向下取10秒整数倍：115.2秒记录使用前110秒，38.4秒记录不使用；不复制成120秒。
- 50秒条件、中间10秒输出、网络不变。训练按每条记录分别归一化损失，两个记录梯度等权平均，允许长度不同。
- PTB按患者固定seed42约80%／20%划分；训练集中优先均匀抽患者，再抽该患者记录。
- 检查非有限值、平直、5秒常量、极端电压、ADC饱和等，技术阈值和每导联原因写进`audit.csv/json`；这是保守技术QC，不代替临床质量认证。
- 坏目标导联仅作为false mask下的有限占位值，不参加损失；坏I记录不进入公开训练。
- 缺失人口学信息按现有schema编码；不虚构身高、体重。
- 公开训练／验证去除精确重复波形。若少于20个训练患者或5个公开验证患者，停止，先看审计结果。

成功标志：`public/public.json`存在且`completed=true`。重复执行会拒绝覆盖；成功后不要重跑。失败保留审计，在新目录修正后重做，不盲目删除已有数据。

## 5. 一次性绑定三组共享study与预检（CPU，不训练）

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_public_slow

run=/home/qht/huawei/results/B5-slow-public/CMP_v1

python -u -m baselines.B5_public_slow.prepare \
  --huawei-cache /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache \
  --public "$run/public" \
  --output "$run/shared" \
  2>&1 | tee "$run/preflight.log"

cat "$run/shared/preflight.json"
)
```

默认从B5-best配置读取SHA固定的当前双模型包。若明确换成另一份已验证的慢CNN起点，可以给prepare加`--bundle /绝对路径/组合包.pt`，且PTB处理也传同一包；FM必须仍为control100。

预检会校验：

1. 缓存来源study的哈希、组件包、control100参数身份与尺度；目录名相同不能代替检查。
2. 所有训练／验证缓存文件哈希、人口学形状、慢标签定义及尺度。
3. 华为和PTB各自训练／验证患者不重叠；华为训练目标去重。
4. 相同验证目标的I、人口学、真实波形及QC一致，保留每个任务配对的FM结果。
5. 可微训练S50前向与部署S50前向完全一致；没有训练或FM生成。

正常应看到`ready: true`、`training_started: false`、`native_max_abs_difference: 0.0`。

如果提示`Cannot prove cache identity`，需提供该control100缓存原来的study文件，不是旧原FM的study。查找命令：

```bash
find /home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1 \
  -name study.json -print
```

然后对上面的prepare命令补充`--cache-study /实际路径/study.json`。程序会核对它的SHA与`cache/cache.json`中的`study_sha256`，不会仅凭名字接受。找不到来源或身份不匹配，先修复缓存来源，不跳过校验。

study成功后无需再次prepare。现有缓存只读复用，不再训练或生成FM。

## 6. 三张卡分别启动C、M、P

打开三个终端，激活BioFlow并进入新工作树。每条命令都会先跑阶段一，再自动跑阶段二，完成后导出可直接推理的`model.pt`。

终端一，GPU0：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow

bash scripts/run_b5_public_slow.sh --group C --gpu 0
```

终端二，GPU1：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow

bash scripts/run_b5_public_slow.sh --group M --gpu 1
```

终端三，GPU2：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_public_slow

bash scripts/run_b5_public_slow.sh --group P --gpu 2
```

日志分别在`CMP_v1/C.log`、`M.log`、`P.log`，不在新组输出内提前创建。脚本通过flock防止同组重复启动。启动后先在CPU校验全部缓存哈希，期间GPU可能暂时没活动；校验完成后才开始慢CNN。Heun16/K16已用于固定FM缓存，慢CNN训练不会每步再运行FM。

每组起点评分应一致；阶段一评分用于诊断，不进入本轮best筛选。阶段二best也允许保留训练前起点，因此best选中stage0不代表训练没启动，要同时看history及completed。

如果要改1560步预算，须在步骤5为三组统一传`--stage1-updates`、`--stage2-updates`后创建新study，不能只改一组或训练中途改。

## 7. 中断后续跑

保持相同提交、study和数据，给原组命令加`--resume`。例如P组：

```bash
cd /home/qht/huawei_B5_public_slow
bash scripts/run_b5_public_slow.sh --group P --gpu 2 --resume
```

它恢复`last.pt`中的网络、优化器、随机状态、抽样位置和计数；若阶段一已经完成，自动进入阶段二。不会重新生成PTB缓存或control100缓存。

## 8. 全部完成后汇总

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_public_slow
run=/home/qht/huawei/results/B5-slow-public/CMP_v1

python -m baselines.B5_public_slow.compare \
  --root "$run" --output "$run/comparison"

cat "$run/comparison/summary.csv"
)
```

报告会同时列出起点、C/M/P的Task1、Task2、平均r和相对提升。若选中stage0，则该组没有刷新起点。三组初始化或评分协议不一致、某组没完成，汇总会停止。

导出结果位置：

```text
/home/qht/huawei/results/B5-slow-public/CMP_v1/C/model.pt
/home/qht/huawei/results/B5-slow-public/CMP_v1/M/model.pt
/home/qht/huawei/results/B5-slow-public/CMP_v1/P/model.pt
```

model.pt内含固定FM、所选慢CNN、尺度及schema，推理不需要训练缓存、公开记录或真实目标。

## 9. 可选：最佳组完整生成复核

先看汇总，下面以P胜出为例。该步骤重新运行FM完整生成，耗时比缓存组合验证长；每个新候选正式部署前复核一次即可。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_public_slow
run=/home/qht/huawei/results/B5-slow-public/CMP_v1
config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml

CUDA_VISIBLE_DEVICES=2 python -u -m baselines.B5_best.validate \
  --config "$config" \
  --checkpoint "$run/P/model.pt" \
  --output-dir "$run/P/deployment_check" \
  --device cuda --verify-reference \
  2>&1 | tee "$run/P_deployment_check.log"
)
```

缓存保留的历史配对FM数值可能与新完整生成有微小差异；若reference不匹配，保留报告检查差值、身份与数据，不能直接放宽容差后宣称复现。

## 10. 打包结果给分析

```bash
run=/home/qht/huawei/results/B5-slow-public/CMP_v1

tar --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  -czf /home/qht/huawei/B5_public_slow_CMP_reports.tar.gz \
  -C "$run" \
  shared public/audit.json public/audit.csv public/public.json \
  C M P comparison C.log M.log P.log preflight.log public_prepare.log

ls -lh /home/qht/huawei/B5_public_slow_CMP_reports.tar.gz
```

最终服务器文件：`/home/qht/huawei/B5_public_slow_CMP_reports.tar.gz`。

## 11. 本地验证范围

新增CPU合成测试覆盖电压单位转换、抗混叠、115.2秒记录适配、真实WFDB文件到缓存、S50前向一致性、mask、来源抽样、两阶段衔接、中途打断后精确续跑、可靠I评分及导出。共36项新增及B5-best回归检查通过，Shell语法检查通过。没有在本地训练真实ECG，也没有验证真实PTB/Huawei效果或GPU显存。

实际文件头示例：[115.2秒记录](https://physionet.org/files/ptbdb/1.0.0/patient002/s0015lre.hea)、[38.4秒记录](https://physionet.org/files/ptbdb/1.0.0/patient001/s0010_re.hea)。因此实现没有采用旧设计中“公开记录必须满120秒”的限制。
