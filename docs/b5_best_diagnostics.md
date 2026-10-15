# B5-best：对已有B5/S50-F预测做诊断，不需要B5-best训练

本工具针对当前最高r的输出组合：`原B5快残差＋S50-F预测慢曲线`。诊断快慢强度、慢预测夹带的快速振荡，以及V1/V3/V5/V6局部形态。只用已有评估数组，不读权重，不启动推理或训练，不占GPU。

## 1. 代码在哪里，数据在哪里

新增代码在本机B5-best工作树：

```text
C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best\
    baselines/B5_best/diagnostics_core.py
    baselines/B5_best/diagnose.py
    scripts/run_b5_best_diagnostics.sh
    docs/b5_best_diagnostics.md
```

`C:\Users\Ashley\Desktop\HW\huawei`当前是main，不要把诊断代码误放成修改main。通过B5-best分支同步，或把上面两个Python文件和脚本按相同相对路径上传到服务器的B5-best代码目录。它们不修改B5-S50训练代码。

也已提供手动上传包`B5_best_diagnostics_code.zip`。假设上传到`/home/qht/`，B5-best代码在独立目录，解压命令：

```bash
python -m zipfile -e /home/qht/B5_best_diagnostics_code.zip /home/qht/huawei_B5_best
```

包内只有新增诊断、测试、脚本和说明，不含模型或数据；服务器需要已有B5-best代码。若代码实际在`/home/qht/huawei`，把解压目标改为该目录，保持相对目录结构。

默认读取以下真实来源，而不是虚构B5-best训练记录：

| 用途 | 服务器位置 |
|---|---|
| 原B5评估根目录 | 从`/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared/prepared.json`的`evaluation_dir`读取，不猜测目录后缀 |
| S50-F评估根目录 | `/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/evaluation_best_score` |
| S50-F组合预测 | 上述根目录下`learned_slow_hybrid/task1`和`task2/prediction_uV.npy` |
| QC mask与对应元数据 | `/home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2/control/evaluations/best_r/task1`及`task2` |
| 已完成三组接入结果 | 同一`fm_slow_qc_three_v2`下的`control/slow/slow_f/evaluations/best_r` |

原B5来源是之前指定第45轮best.pt对应的评估。代码不读取.pt，只校验已有prepared和评估报告中的来源SHA，以及原始评估文件SHA。QC目录只有`target_reference.json`而没有复制目标数组时，也会正确追溯原目标文件。

本机分析用的tar包没有大预测数组，不能只上传报告包运行；需要服务器保留此前实际评估生成的.npy文件。若文件被清理，工具会明确报缺失路径，不会悄悄启动GPU重新生成。

## 2. 一条命令运行

若服务器B5-best代码在之前约定的独立目录：

```bash
cd /home/qht/huawei_B5_best
conda activate BioFlow
sed -i 's/\r$//' scripts/run_b5_best_diagnostics.sh
bash scripts/run_b5_best_diagnostics.sh --repository-root /home/qht/huawei
```

若B5-best代码实际就在`/home/qht/huawei`，只把第一行改为`cd /home/qht/huawei`。`--repository-root`仍表示原数据/结果根目录。不要为本诊断切换正在训练的代码目录。

**不用CUDA_VISIBLE_DEVICES，不需要选择GPU。** CPU读盘校验和计算期间GPU不动是正常的。

脚本打印本次结果路径，例如：

```text
/home/qht/huawei/results/B5-best/diagnostics_20261015_230000_12345/
```

它只是在B5-best名字下存新诊断报告，不表示需要已有B5-best训练。日志在该目录旁的同名`.log`，不影响“输出必须为空”的保护。

成功条件：`diagnostics.json`中`completed:true`、`training_started:false`、`inference_started:false`；beta=1重算r应与当前参考0.69585536/0.68940334一致，允许2e-5数值差。

## 3. 路径有差异时显式指定，不改权重

例如服务器三组结果目录不同，或只有slow组评估可提供QC：

```bash
cd /home/qht/huawei_B5_best

python -u -m baselines.B5_best.diagnose \
  --repository-root /home/qht/huawei \
  --integration-root /home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2 \
  --qc-evaluation-dir /home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2/slow/evaluations/best_r \
  --output /home/qht/huawei/results/B5-best/diagnostics_manual_01
```

这只从slow评估读取QC和目标身份，诊断的基准FM仍是prepared绑定的原权重预测，不会误换成slow组预测。

其他可覆盖项：`--prepared-info`、`--fm-evaluation-dir`、`--slow-evaluation-dir`。最后一个参数要传`evaluation_best_score`根目录，不传.pt，也不传其`learned_slow_hybrid`子目录。

想先只核对路径、SHA、目标和QC及基准r，可给上面的命令加`--check-only`。这也需要新输出目录；正式诊断换另一个新目录，不覆盖预检报告。

## 4. 输出分别回答什么

### A. 快慢强度与振荡

- `beta_scores.csv`：统一比较`快残差＋0.75/1/1.25×预测慢曲线`，用原完整120秒、导联QC、目标等权、Task2设备等权评分。
- `component_record_lead.csv`：逐唯一目标正常导联的慢/快标准差比、各分量r、慢曲线超过1Hz的能量占比和目标匹配残差误差。
- `component_lead_summary.csv`：按导联汇总，优先看V5/V6。

beta改变的是相对慢起伏强度，不是整条波形统一放大。三点只是预先固定的验证探针，程序不按每条记录挑系数，不自动将最高探针晋升为正式模型。若有信号，之后在训练集定义规则或做独立复核。

501点均值不是理想1Hz截止滤波，真实慢标签也可能含1Hz以上成分。应比较预测和真实标签，不能看到高频能量就认定全是错误或全部去掉。

### B. 局部形态

- `morphology_record_lead.csv`：对相同I锚点比较FM与组合后的快曲线，记录峰谷幅度比、峰值时间差、半高宽差、未平移局部r等。
- `morphology_lead_summary.csv`：汇总V1/V3/V5/V6。
- `qrs_anchor_audit.csv`：每条记录的I锚点候选数量。算法只给近似QRS候选，不是临床标注；明显漏检/多检的记录需要看图，不可直接据此判断模型心搏错误。

I锚点来自同步真实I，即任务允许观察的输入导联；其他真实导联只用作诊断标签。局部窗口是固定的[-100,+120]ms范围，不声称已经精确划定QRS或T波。正/负峰方向依据真值仅作诊断。

`best_lag_DIAGNOSTIC_ONLY`允许局部±20ms寻找对应关系，只用于看是否存在时间偏移；不平移正式输出、不改变正式r。未平移局部r才是原始局部对照。

### C. 边界与已有接入结果

- `boundary_record_lead.csv / boundary_lead_summary.csv`：看边界附近是否比窗口中部误差集中，不把真实波形跳变直接当接缝。
- `integration_scores.csv`：核对已有B5-S50三组的同QC纯r；若某组目录缺失，标记缺失，不重新推理。
- `plots/`：低/中/高诊断r的代表记录；原始局部、快残差局部、慢走势、同I锚点叠加与慢曲线频谱。慢走势显示面板单独去均值只为了看形状，不影响正式输出。

单条目标诊断优先使用机器配对，避免把同一目标重复算成多个独立患者。它与正式两任务分数目的不同，不用诊断均值替换正式r。

## 5. 发哪些文件分析

把脚本最后打印的目录填入`run`，无需打包权重、预测数组、原始数据：

```bash
cd /home/qht/huawei
run=results/B5-best/diagnostics_这里替换为本次实际目录名
tar -czf B5_best_diagnostics.tar.gz "$run" "${run}.log"
```

如果用了手动Python命令且没保存同名log，最后一项删去即可。诊断目录本身只保留轻量CSV/JSON/Markdown/PNG；临时beta评分数组成功后自动清理。

## 6. 本地验证范围

本地新增诊断10项与原B5-best回归10项，共20项CPU检查通过，Bash语法检查通过。没有运行真实ECG推理或训练。重点验证：来源路径、窗口顺序/身份对齐、QC坏导联排除、beta=1对同来源数组重算一致、比例与已知时间偏差诊断、缺失I候选处理、target_reference追溯，以及不依赖B5-best训练目录的完整文件流程。真实服务器文件的可用性和历史r由运行时预检核对。
