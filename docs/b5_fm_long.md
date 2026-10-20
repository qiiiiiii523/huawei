# B5-final-exp-FM 10秒与40秒完整波形生成实验

本分支从 B5-final 的 `9f3e962` 创建，独立工作树为
`C:/Users/Ashley/Downloads/备赛/B5-CMF工作树/Final-FM`。
不修改原 B5-final、slow120、directfast、原权重或正式提交接口。

## 1 本轮回答什么

FM仍学习完整波形；只改变同步I与FM联合生成窗口的长度。不是新增FM慢分支。

| FM窗口 | raw | hybrid |
|---|---|---|
| 10秒 | FM完整原始输出 | FM快残差＋0.75×原慢CNN |
| 40秒 | FM完整原始输出 | FM快残差＋0.75×同一原慢CNN |

两个输出共享同一次FM生成。40秒按0～40、40～80、80～120秒三段生成；没有尾段补齐、重叠平均或目标padding。
最终仍按完整120秒、可靠同步I、逐目标导联QC、目标记录等权、Task2设备等权、两任务平均r评价。
首轮**关闭设备context**；不能拿无context结果直接与完整Final的0.707514相减。
当前Final无context的历史均值约0.705557；只有数据/QC完全相同时才可作复现参照。

## 2 两阶段执行

1. 默认只准备数据并用现有FM权重做10秒/40秒推理筛查，优化器更新为0。
2. 显式加`--train`才启动两组匹配续训。第0步参与raw和hybrid各自选模。

直接长窗推理是分布变化诊断，不等同于训练过长窗口；零训练筛查失败不能证明长窗口训练无效。
GroupNorm窗口统计、边界和ODE迭代的信息传播也会随窗口改变，不能把差异全部归因于远距离信息。

## 3 固定内容与配对方式

| 项目 | 设置 |
|---|---|
| 初始化 | 当前Final的完整base包，SHA为`990f5d95312dba0ab3e4bde416ee15d3b464b902fc63601aab1f852d9881f6f8`；取FM EMA |
| 网络 | 原64/128/256条件U-Net、同步I多尺度注入、人口学FiLM |
| 训练监督 | 原完整波形FM速度损失及原辅助损失；不改为快标签训练 |
| 慢CNN | 原包内慢CNN冻结；每条记录仅准备一次慢预测 |
| 正式采样 | Heun16/K16、seed42 |
| 推理噪声 | 先按原`记录ID:10秒起点`规则生成10秒噪声块；40秒连接相同四块，保证相同物理点使用相同噪声 |
| 训练曝光 | 每次更新抽相同的2条记录和相同40秒片段；FM10拆成8个10秒微批，FM40做2个40秒微批 |
| 训练随机量 | 两组共享记录、片段、路径噪声及每个40秒块的生成时间；条件dropout调用次数不同，不声称完全相同 |
| 默认试验预算 | 390次更新，每78次验证；每组累计31200秒波形曝光，均为同一训练患者集的重复使用 |
| 优化器 | 新AdamW，lr=1e-5，weight_decay=0.01，EMA=0.999，梯度限幅1；全精度 |
| 选模 | 独立保存`best_raw.pt`和`best_hybrid.pt`，另保存`last.pt`；慢CNN从不更新 |

同样更新次数、同样真实信号曝光，不代表墙钟时间相同；输出记录运行耗时。
本轮只检验现有骨干上的长窗口生成，不声称已增加全局注意力或证明FM40是最优长度。

## 4 服务器首次建立工作树

先将本地分支自行推送并同步到服务器可访问的远程。服务器示例：

```bash
cd /home/qht/huawei
git fetch gitee B5-final-exp-FM
git worktree add /home/qht/huawei_B5_final_exp_FM gitee/B5-final-exp-FM
cd /home/qht/huawei_B5_final_exp_FM
conda activate BioFlow
```

工作树若已存在，先检查本地改动，再用现有流程更新；不要重复执行worktree add或覆盖其他实验。
服务器继续从`/home/qht/huawei`读取原数据、结果和选定权重，不搬动它们。

## 5 默认先做不训练的筛查

```bash
cd /home/qht/huawei_B5_final_exp_FM
bash scripts/run_b5_fm_long.sh --gpu 1
```

默认创建全新`/home/qht/huawei/results/B5-final-exp-FM/run_时间_PID/`。
脚本会准备完整训练/验证记录，固定原始目标、QC、人口学、尺度和慢预测，跑四格并生成`comparison/summary.md`。
准备阶段需要现有两个任务的完整缓存、严格训练索引、患者划分和人口学；不生成训练集FM预测。
无效输入I的验证记录仍保留在缓存，评分器按原规则排除；不因历史缺失或模型误差删记录。

权重搬迁后：

```bash
bash scripts/run_b5_fm_long.sh --gpu 1 \
  --checkpoint /实际位置/base.pt \
  --config /home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
```

不需要body.pt。可通过`--fm-source-checkpoint /原FM训练检查点`从原训练权重读取损失；必须匹配base包记录的源SHA。
未提供时使用已有Final数据run配置中的原损失，来源及SHA写入study。
源码、配置、完整记录缓存和权重身份均绑定；成功准备后不要改代码再续用旧study。
筛查不自动宣称复现完整Final或获得性能提升，请核对FM10/hybrid与同口径无context基准。

## 6 再决定是否训练

一次完成准备、筛查、两组训练、独立复评与推理包核对：

```bash
bash scripts/run_b5_fm_long.sh --gpu 1 --train
```

已有准备目录可复用到**全新的**结果目录（不会重复复制完整数据）：

```bash
bash scripts/run_b5_fm_long.sh --gpu 1 --train \
  --shared /home/qht/huawei/results/B5-final-exp-FM/前一运行/shared
```

`--shared`时以该study保存的实验配置为准，不再读取命令行`--experiment-config`来改变旧study。
若调整预算或超参数，应在prepare前复制并修改`configs/b5_fm_long.json`，用新目录准备一份study；两组共用它。

中断后，从该组最后完成验证时保存的检查点恢复，例如：

```bash
python -u -m baselines.B5_fm_long.train \
  --run /实际运行/shared --window-seconds 40 --output /实际运行/FM40 \
  --device cuda --batch-size 1 --execute-training --resume
```

默认最多回退77次未保存更新，按保存的优化器和RNG重新计算；不覆盖已有评估目录，重算验证使用新attempt目录。
一组一个输出目录，不能并发写同一组。失败保留文件；不要原样重跑一键脚本指向已存在run。
完整一键流程若在后半段中断，按下列单步入口补完未完成步骤，并使用未存在的输出目录。

```bash
# 独立复评同一候选，输出两个模式
python -m baselines.B5_fm_long.evaluate --run /实际运行/shared \
  --candidate /实际运行/FM40/best_raw.pt --output /全新复评目录 --device cuda

# 导出不含优化器、RNG或训练波形的独立推理包
python -m baselines.B5_fm_long.predict package --run /实际运行/shared \
  --candidate /实际运行/FM40/best_raw.pt --output /全新位置/FM40_raw.pt

# 不依赖study缓存，由可见I重新计算FM和慢CNN
python -m baselines.B5_fm_long.predict predict --package /全新位置/FM40_raw.pt \
  --anchor-npy /实际输入/current_I_uV.npy --metadata-json /实际输入/demographics.json \
  --record-key 实际稳定记录ID --mode both --output /全新预测目录 --device cuda
```

输入I为原始μV、500Hz、完整120秒。独立包仍需要本分支代码和原环境，但不需要原分散权重、训练/验证缓存或隐藏目标。
这是实验包，不是正式B5_final包；不要交给原正式提交入口自动替换使用。

## 7 怎样判断结果

| 现象 | 当前能得到的结论 |
|---|---|
| FM40 raw超过匹配FM10 raw | 扩大生成窗口对FM完整输出有帮助 |
| FM40 hybrid也超过匹配FM10 hybrid | 长窗变化能进一步帮助现有快慢组合 |
| 仅raw改善，hybrid不改善 | 可能主要改善了最终被替换的慢部分，需要快/慢与边界诊断 |
| FM40 raw超过当前固定组合，且hybrid不再受益 | 有理由继续检验是否可移除慢CNN，仍需训练seed与逐患者复核 |
| 两种模式均未超过原始和匹配对照 | 本轮没有形成升级，不自动扩大轮数或扫描更多长度 |

分别看Task1、Task2两设备和V1/V3/V5/V6等逐导联变化；同一患者重复配对不当作独立样本。
`component_diagnostics.csv`在生成后按有效目标导联记录完整/慢/快r、慢幅度比例，以及相同10秒边界位置的跳变误差；它们仅作离线诊断，不替代正式r，也不会进入预测器。
训练best、末步和第0步全部保留。脚本会复评两种best与last，并从推理包重新生成，核验选模与推理一致。
默认保留评估数组供慢分解、边界和患者诊断，完整训练流程需预留数十GB，实际大小取决于验证配对数。
FM窗口变化会改变body输入分布；本轮关闭context的结论不能直接当作完整部署收益。有效后另做冻结context迁移/适配。

## 8 本地功能检查

```bash
python -m unittest discover -s baselines/B5_fm_long/tests -v
python -m unittest discover -s baselines/B5/tests -v
```

本地CPU合成检查验证实现合同，不代表真实ECG性能；真实数据筛查和训练应在服务器按上述步骤执行。

2026-10-20本地检查：新增14项功能测试通过；连同原B5、组合推理及Final共118项回归中117项通过，1项真实缓存检查因未配置环境跳过；Bash语法检查通过。检查覆盖噪声配对、原10秒采样兼容、40秒真实长度、QC/患者隔离、缓存篡改拒绝、合成优化与精确恢复、候选导出及独立包加载。未在本机启动真实ECG训练或生成新的性能成绩。
