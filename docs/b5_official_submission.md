# B5-final：主办方测试接口和Windows提交

本次只适配官方测试流程，不重新训练、不换权重、不改变已选模型。当前代码仍在 `B5-final`，本地目录 `C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Final`。

## 1. 固定模型与官方入口

沿用 `configs/b5_final.json` 固定的FM-best45＋P慢CNN、β0.75、Heun16/K16/seed42、机器d6 FIR25、body seed43-best3 II适配。两份权重SHA仍严格核验，不按文件名寻找最新best。

新增入口：`python competition_main.py`，EXE名为 `B5Submit.exe`。两种运行都以程序文件所在目录为基准，而不是终端的工作目录，默认读取同级 `test_data`，写同级 `results`。

```text
提交目录/
├── B5Submit.exe
├── _internal/           # PyInstaller依赖，必须与exe一起保留
├── models/
│   ├── base.pt          # 完整FM＋P推理bundle
│   └── body.pt          # 选定body修正器
├── submission.json      # 运行配置
├── test_data/           # 主办方放入，不随提交ZIP发送
│   ├── userinfo.txt
│   ├── watch/被试编号/*.txt
│   ├── scale/*.csv
│   └── Electrocardiograph/
│       ├── d12_test/被试编号.xml
│       └── d6_test/被试编号.xml
└── results/             # 本轮运行生成
    ├── 被试编号.csv
    └── _audit/          # 来源、任务回退、输出SHA；参考评分只读取顶层CSV
```

通过未修改的主办方 `input.py` 逐人读取电压与人口学；通过未修改的 `output.py` 写入并重新读取校验CSV。原版文件保存在 `competition_interface/`，SHA见 `source_manifest.json`；官方4点虚构样例也保留在那里用于格式检查。

**主办方提供的sample只有4点，不是120秒测试信号。** 完整推理拒绝把它重复/补齐成60000点；请用 `--inspect-input` 测试读取，不要把格式样例的无法推理误判为模型错误。

## 2. 三个任务如何映射

|官方输出|当前anchor|历史条件|保持的模型路径|
|---|---|---|---|
|gen1|ele.d12.I|当前选定模型不使用watch|原FM＋P完整输出|
|gen6_scale|同一ele.d12.I|scale原始通道1,2,9,10,11,12，映射I,II,III,aVR,aVL,aVF|原body II修正；内部执行一次B去漂移|
|gen6_d6|同一ele.d12.I|ele.d6的六肢体导联，按实际采样率转500Hz|原d6投影＋FIR25|

三任务使用相同anchor、人口学、稳定 `被试ID:当前记录ID` 噪声key。基础模型只运行一次，再调用原ContextRouter分别产生三份结果，算法与原单任务入口一致。历史不与当前120秒按时间点拼成同步信号，不读取训练缓存、隐藏胸导联或金标准。watch仍由官方接口读取，但本选定版本保持不使用watch修正的既有设置。

官方输入已经转为μV，部署层不重复执行XML scale/origin或体脂秤单位换算。额外解析原文件仅取得采样率、Index位置和显式输入导联质量声明。XML只读RHYTHM时序；体脂秤Index间隙恢复为无效mask，不把缺口前后挤在一起；降采样在连续有效段内完成。

不额外调用主办方可选 `preprocess.py`，不改变anchor基线、不对输出再减均值、不再次放大电压。body收到的是raw历史，在其既有适配器内执行一次B处理。

每任务输出 `[12,60000]`、500Hz、μV，导联顺序固定 I、II、III、aVR、aVL、aVF、V1～V6。每人一个39列CSV。初赛按官方r参考程序评估，RMSE仅记录，不在提交入口额外合成60分。

## 3. 输入异常与多记录策略

- 当前I必须包含至少120秒真实有限信号；取开始120秒，按实际采样率转换为60000点，不padding。不存在可用I时整轮报错，不伪造答案或跳过该被试。
- 每人如有多条当前I，选择最长的可用记录，时长相同按record_id字典序。历史各设备分别选择有效I/II时长最多的记录，同分按record_id；不拼接不同次采集。所有选择写审计。当前官方目录通常每人一个d12/d6 XML；此策略使多条scale记录可确定复现。
- 机器历史读取显式“信号质量差（导联）”声明，只mask真实可见的肢体导联；预期缺少胸导联不会使六肢体历史失效。体脂秤使用有限、非恒定点与Index缺口mask，没有测试真值QC可读。
- 缺少/过短/不可用历史，沿用已有模型回退：机器仅一致性投影，body通用II修正；仍输出三个任务并记录回退。不会将缺历史当作读取真值的理由。
- 采样率优先取XML的TIME_ABSOLUTE/TIME_RELATIVE increment或scale文件头。**不能默认输入500Hz**，原心电图机常见1000Hz。无采样率的正式文件需要先核实，再在 `submission.json` 的 `d12_fs_fallback`、`d6_fs_fallback` 或 `scale_fs_fallback` 中显式填写；默认null，禁止从数组长度猜采样率。4点样例d12恰好没有该元数据，但只运行inspect无需填写。
- 输出目录必须为空。二次测试应先归档旧results；不混入旧人、不追加CSV。失败保留已完成文件和审计，下次需新目录。

## 4. 现在服务器上先做Python真实权重自测

更新到包含本次提交的B5-final代码。若服务器工作树为 `/home/qht/huawei_B5_final`，进入该目录：

```bash
cd /home/qht/huawei_B5_final
conda activate BioFlow

python competition_main.py \
  --data-root competition_interface/test_data \
  --inspect-input
```

预期1人，各设备1条，d12长度4；这一步不加载模型、不写results，也不是性能测试。

检查两份真实模型能否加载，使用原服务器路径：

```bash
CUDA_VISIBLE_DEVICES=1 python competition_main.py \
  --repository-root /home/qht/huawei \
  --checkpoint results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --body-checkpoint results/B5-best-context/newFM_body_adapt_v1/adapt/best.pt \
  --device cuda \
  --check-model
```

检查通过仅代表权重、配置与架构加载成功，不是前向/成绩复现。随后把一条/数条**真实且测试可见**的输入按官方目录放到 `/home/qht/official_test_data`，不能把完整12导联真值复制进去；d12_test仅保留I。运行：

```bash
CUDA_VISIBLE_DEVICES=1 python competition_main.py \
  --repository-root /home/qht/huawei \
  --checkpoint results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --body-checkpoint results/B5-best-context/newFM_body_adapt_v1/adapt/best.pt \
  --data-root /home/qht/official_test_data \
  --output-dir /home/qht/B5_final_official_results_v1 \
  --device cuda
```

输入文件由官方接口解析、三任务预测、原版CSV保存与读回校验全部通过后，检查 `results/_audit/completed.json`。`record_key`与历史内部验证ID不同会改变固定噪声，不能仅凭官方样例入口要求复现旧报告的r到1e-7。

如果要制作可搬迁Python运行包（可先在服务器验明完全脱离训练目录）：

```bash
python -m baselines.B5_final.package \
  --base-checkpoint /home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt \
  --body-checkpoint /home/qht/huawei/results/B5-best-context/newFM_body_adapt_v1/adapt/best.pt \
  --destination /home/qht/B5_submission_python
```

该ZIP是Python自测包，不是最终Windows EXE。将真实test_data放入包目录后，在任意工作目录运行 `python /home/qht/B5_submission_python/competition_main.py`；不需再指定服务器权重路径。包内没有训练数据、金标准或真实test_data。

## 5. 在Windows打包真正EXE

Linux服务器不能直接编出Windows EXE。把选定的两份原始.pt复制到Windows（不能只下载之前的reports压缩包）。例如放到 `C:\B5Weights\base.pt` 和 `C:\B5Weights\body.pt`，内容SHA必须与固定模型一致。

使用独立Windows Python 3.10/3.11环境；先装运行/构建依赖。建议使用已验证能加载该权重的PyTorch版本；GPU轮子需与测试机相符，CPU环境同样可以运行但K16会慢。主办方没有在附件规定GPU型号/时限，应按实际环境自测。

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Final'

python -m pip install -r requirements-submission.txt

python -m baselines.B5_final.package `
  --base-checkpoint C:\B5Weights\base.pt `
  --body-checkpoint C:\B5Weights\body.pt `
  --destination C:\B5Release\B5Submission `
  --build-exe
```

输出：`C:\B5Release\B5Submission\B5Submit.exe` 及同级依赖、两份models、配置，ZIP为 `C:\B5Release\B5Submission_package.zip`。必须提交整个包，不能只取exe。此命令不联网下载模型、不训练、不改变权重；安装依赖是显式pip步骤。

把真实test_data放到EXE同级后，先执行：

```powershell
C:\B5Release\B5Submission\B5Submit.exe --check-model
C:\B5Release\B5Submission\B5Submit.exe
```

EXE无需参数就按主办方目录运行全部被试。默认auto选择可用CUDA，否则CPU；可在同级submission.json设置device和batch_size。显存不足时减batch_size（不改K16或权重）。错误会非零退出并写同级 `submission_error.log`，不要把失败的部分CSV当成完整提交结果。

## 6. 官方参考评分（只有本地有合法金标准才能运行）

```bash
python -m baselines.B5_final.official_score \
  --pred-dir /home/qht/B5_final_official_results_v1 \
  --gold-dir /path/to/heldout_gold_csv \
  --report /home/qht/B5_official_reference_score.json
```

调用原版evaluate.py，保持其相关性权重、恒定信号r=0、RMSE算法；全部11个待生成导联参与评分，不应用本地目标QC过滤，I不计分。r分数用于初赛；原版结果中的RMSE是诊断，不自动加bonus。预测函数不接受金标准，gold仅存在于这个离线评分入口。

## 7. 当前完成与待验证边界

本地有代码和官方4点样例，**没有选定两份真实权重，也没有正式隐藏测试数据**。本次未训练、未执行真实GPU推理、未生成声称可用的空壳EXE。已提供可执行构建流程，实际二进制仍需取得两份权重、在Windows构建后进行干净电脑运行验证。

19项新增及既有回归检查通过：涵盖官方文件SHA、4点样例只读、XML单位/1000→500Hz、真实120秒长度、scale通道映射与缺口、显式输入QC、历史回退、三任务共享基础预测且结果等价、实际CSV60000行39列读回、原版参考评分、EXE相对路径、Python包搬迁，以及既有Final权重固定与路由。合成端到端用预测替身验证文件与路由，不是模型成绩。正式权重与干净Windows环境尚待验证。
