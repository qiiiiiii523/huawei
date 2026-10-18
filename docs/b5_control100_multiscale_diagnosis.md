# 第0轮、B20、C20诊断与并行复核

## 1. 本次完成的内容

脚本：scripts/diagnose_b5_control100_multiscale.py。
本地分支仍是B5-best-exp-Control100PCC，工作树仍是：

```text
C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Control100PCC
```

本次只新增scripts及文档，不修改baselines或ecg12gen核心。因此不会改变已经冻结的三组study代码指纹。
不用创建新模型分支或worktree，不重训，不重新生成FM，不修改默认部署模型。

诊断顺序：

1. 用已有数组复算第0轮/B20/C20完整120秒正式r，核对新可靠I评分和原报告相符。
2. 每个配对使用同一control100缓存，恢复实际慢贡献：最终预测−(固定FM−完整120秒MA501固定FM)。
3. 真实慢仍是完整目标的MA501；不是把最终预测平滑后冒充独立慢CNN。
4. 按约<5秒、5～15秒、>15秒尺度比较相关性、起伏强度和形状误差。
5. 逐导联、逐目标及逐患者比较C−B、C−第0轮。
6. 按患者成组bootstrap复核平均r差异，可选生成6张重点案例图。

程序只需要numpy；画图另需matplotlib。无需torch、模型权重、scipy或GPU。它读取验证缓存，不读取训练波形。
只有验证标签/目标均值用于离线诊断；不调整任何预测，不将真值、均值或平移提供给模型。

## 2. 诊断和复核能否并行

可以，但要区分两类复核：

|工作|本次是否有入口|资源|作用|
|---|---|---|---|
|分尺度、患者、导联诊断|有|CPU|确定变化发生在哪些记录/尺度|
|患者成组bootstrap|有，与诊断一起完成|CPU|估计固定模型在当前患者集合上分数差异的不确定性|
|B/C另一训练seed复核|本次不新增训练入口|GPU|检验换训练随机性之后结论是否仍成立|

诊断可以与独立训练seed复核同时进行，前提是后者用新study和独立输出目录，不覆盖本轮第0轮/B20/C20及公共缓存。
CPU诊断/画图限制线程后，不占GPU，但仍会使用磁盘和CPU。bootstrap并不代替训练seed复核。
现有prepare固定训练seed42，不能只在旧训练命令后加--seed 43；本次脚本的--seed只控制bootstrap抽样。
如果后续做训练seed复核，需要显式增加可审计的训练seed配置；如果延长预算，也不能直接编辑已冻结study的epochs。
本次不会自动启动任何训练。

## 3. 本机推送与服务器更新

本机PowerShell：

```powershell
cd "C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Control100PCC"
git push origin B5-best-exp-Control100PCC
```

origin是GitHub；按原有镜像流程将分支同步到Gitee后，在服务器执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_control100_pcc_ms

git status --short
git fetch gitee B5-best-exp-Control100PCC
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

如果手动上传，上传新脚本到：

```text
/home/qht/huawei_B5_control100_pcc_ms/scripts/diagnose_b5_control100_multiscale.py
```

本脚本复用该工作树现有numpy工具和主评分接口，不要上传其他旧分支的baselines/ecg12gen文件来覆盖它。
已有本地修改导致更新失败时保留文件，不执行reset --hard。

## 4. 运行诊断、画图、bootstrap并自动打包

可在现有三组训练结束后立即执行。以下命令会自动选一个新的输出目录，日志在目录外，不会造成输出非空错误：

```bash
conda activate BioFlow

(
set -euo pipefail
cd /home/qht/huawei_B5_control100_pcc_ms

parent=/home/qht/huawei/results/B5-best-exp-Control100PCC-diagnostics
mkdir -p "$parent"
out="$parent/diag_$(date +%Y%m%d_%H%M%S)_$$"

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u scripts/diagnose_b5_control100_multiscale.py \
  --run /home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1 \
  --output "$out" \
  --bootstrap 2000 \
  --seed 42 \
  --plots \
  --max-plots 6 \
  2>&1 | tee "${out}.log"

tar -czf /home/qht/huawei/B5_control100_multiscale_diagnostics.tar.gz \
  -C "$parent" "$(basename "$out")"

echo "诊断目录：$out"
echo "下载：/home/qht/huawei/B5_control100_multiscale_diagnostics.tar.gz"
)
```

成功标志：最后completed=true；diagnostic.json的completed=true、training_started=false、model_loaded=false、new_FM_inference=false。
显卡不动正常。开头会校验缓存、复算正式分数，耗时取决于磁盘与CPU，终端会持续打印阶段与配对进度。
若缺matplotlib，可先从上述命令删掉--plots及--max-plots 6，其他表格和bootstrap仍完整；不需要为了表格安装画图库。
脚本不覆盖已有诊断目录，重跑使用新时间戳。缺失源数组时明确报错，不能仅靠CSV汇总恢复波形诊断。

默认读取：

```text
第0轮：partner_multiscale_v1/adapt_control100/validation/epoch_0000
B20：  partner_multiscale_v1/adapt_control100/validation/epoch_0020
C20：  partner_multiscale_v1/adapt_multiscale/validation/epoch_0020
FM：   shared/cache_view.json指向的control100逐配对验证缓存
```

当前best恰好是第20轮，因此默认与本次最好结果一致。未来若换成别的轮次，本入口仍要求明确的0/20轮身份，不能把未来best或别的模型偷偷放入C20。
支持--initial-evaluation、--b-evaluation、--c-evaluation指定相同组/轮次的其他评估目录，例如evaluation_best的副本；脚本会核对组、轮次、采样、β、新评分协议和原分数。
来源文件在使用前后均检查SHA，目标、患者、设备、配对和窗口按身份对齐，不按数组行号盲目相减。

## 5. 输出怎么看

|文件|作用|
|---|---|
|score_summary.csv、verification/|三组正式120秒新QC分数复核|
|pair_lead_metrics.csv|各组逐配对、逐导联完整/慢/短/中/长指标|
|target_lead_metrics.csv|先平均同目标的重复配对，避免重复记录占更大权重|
|lead_summary.csv|每个任务/设备/导联的各指标均值、中位数、有效目标数量|
|target_lead_comparison.csv|逐目标逐导联的初始/B/C指标及C−B、C−初始变化|
|target_diagnostics.csv|各评分集合中的逐目标平均r及变化，供定位失败记录|
|patient_diagnostics.csv|同一患者跨任务/设备视图的诊断平均及受益/退步情况|
|patient_bootstrap.json|保持正式宏平均层级的患者成组bootstrap结果|
|qc_exclusions.csv|输入I整条排除、目标坏导联局部排除的明细|
|plots/、plot_selection.csv|最多6张案例图、选例依据、实际配对和所选窗口|
|diagnostic.json、report.md|来源、公式、完成状态和统计解释|

正式r仍是：拼120秒→同目标配对r平均→每个导联对目标等权→导联等权→Task2两设备等权→两任务等权。
逐患者简单平均不等于正式R，尤其各记录的QC正常导联数不同，所以patient_diagnostics仅用于定位，不能代替score_summary。
bootstrap会用逐目标逐导联r重新构建上述正式层级，并与主评分接口核对到1e-10；同一患者的所有记录、任务和设备视图共同抽样。
置信区间只针对已选好的三份模型与当前验证患者，未修正反复选模/调参造成的偏差，不是独立测试保证。
fraction_draws_positive不是p值；区间跨0表示当前患者集合下稳定性证据不足，不等于证明没有效果。
若某次抽样导致某设备/导联没有可评分记录，会报告无效抽样数，不把空值填0。

## 6. 指标与图的边界

- slow_r比较实际β=0.75慢贡献与完整目标MA501；slow_std_ratio_raw_CNN除去β，只作诊断，不在输出中加倍率。
- short/middle/long是实际部署慢曲线在50Hz块平均后，再用251/751点平滑拆分；近似训练尺度，不是直接读取原生CNN输出，也不是理想频率切分。
- 本诊断按正常I/目标导联QC统计全部可定义项，不复用训练标签的近常数门槛；常数标签相关性留空，并报告有效数量与真实起伏强度。
- final_fast_r是最终波形再次做同一MA501拆分后的较快部分，检查慢CNN变化是否影响局部形态；fixed_fast_r在三组应一致。
- 图中第一行保留真实电压位置；三个尺度的均值去除只为显示，不会修改波形或正式评分。
- 10秒窗口仅用于看局部形态，按真值诊断的变化选择示例，不用窗口r代替120秒正式r，不在测试时据此挑片段或校准。

6张图依次尝试选：Task1 V3/V6相对初始提升最大的病例，Task2体脂秤组V1/V6相对初始下降最大的病例，体脂秤组V5相对B改善最大的病例，Task1 V5相对B下降最大的病例。
都必须满足可靠I与该目标导联QC；这是说明性选例，不能把这6张图当作总体表现。

下一步先依据这些结果判断：新增监督改善了哪些尺度、是否对Task2特定患者/导联造成取舍；再决定训练seed复核、延长预算或细化尺度权重，不在本次自动改网络。
