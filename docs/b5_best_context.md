# B5-best-context：机器 FIR25 接入与三组对照

本分支从 `B5-best` 的 `ec12008` 创建。保留当前 control100 FM＋SlowPCC20、beta=0.75、EMA/Heun16/K16，按当前可靠 I＋目标导联 QC、目标记录等权、Task2 两设备等权计算 `(r1+r2)/2`。本轮不训练。

## 1. 本次接了什么

```text
当前I＋人口学 → 原B5-best两网络 → 拼完整120秒并完成快慢组合
                                                ↓ 原始μV
           Task2机器 → 一致性投影 → 75%投影II＋25%个人FIR估计II
                                   → 重建II/III/aVR/aVL/aVF
           Task2体脂秤 → 原输出
           Task1       → 原输出
```

机器历史 I/II 只用于从该条合法历史拟合个人 FIR；历史六通道仍按原公共接口读取。FIR 为61点、500Hz、ridge alpha=0.01、固定25%混合，继承队友已测试方法。不是25点滤波器。

新分支不加载队友 V6 权重，不替换 B5-best 两个网络，不变更模型 scale。只移植独立 NumPy 后处理。V1–V6、输出 I、Task1 和体脂秤默认逐点不变。

新配置：`configs/b5_best_context.json`。机器默认 `fir25`；体脂秤/手表默认 identity。未来体脂秤可实现 `ContextRouter(..., body_adapter=...)` 的目标无关适配接口；当前没有空网络、随机权重或未经验证的体脂秤 FIR。若未来体脂秤是网络内部注入，应另建匹配实验，不假设仅后处理接口足够。

## 2. 数据与回退

- 当前 I：真实可见的120秒同步I，500Hz、原始μV。FIR约束使用观测I，不使用模型预测I。
- 历史：从公共独立记录缓存取原始μV，按历史自身顺序拼接，去掉尾窗padding，保留内部缺口位置。time/window/lead QC mask一起使用。
- 不使用归一化context；不把去median后只乘回scale的信号当原始历史。
- 不读取当前真实II/胸导联拟合个人参数。不按当前验证r调患者混合比例。
- 历史缺失、过短、有效点不足或退化：`fir25`回退**仅投影**，不是原始输出，也不是公共FIR。
- 沿用队友算法：历史前40%/后40%隔离检查用于有限/可计算性检查；不以holdout r高低再筛患者或调权。即使holdout r偏低，也可能启用FIR，报告会如实记录，第一轮不改变方法。

当前I与其他目标在公共验证缓存中同记录，所以评估入口从缓存的I通道取得已知输入；其余目标只在评分/事后诊断使用，不传入router/FIR接口。

## 3. 三组实验与文件

|组|处理|目的|
|---|---|---|
|baseline|原B5-best|固定基准|
|projection|仅机器肢体一致性投影|确定结构约束的收益|
|fir25|投影＋25%个人FIR|相对projection判断历史个人信息的额外作用|

只进行一次原 B5-best 生成，三组复用同一批预测、噪声、人口学和QC。基础缓存可按目标ID共享；**后处理必须按pair/设备执行**，不可把机器增强输出缓存后复用于体脂秤/Task1。

```text
运行目录/
├── base_inference/        # 新生成时保存；复用已有评估时不生成
├── run.json
└── study/
    ├── summary.csv / summary.md
    ├── study.json
    ├── pair_comparison.csv
    ├── baseline/
    ├── projection/
    └── fir25/
```

每组有 `evaluation.json`、`correlation_score.*`、`pair_metrics.csv`、`postprocess_records.csv` 和两任务的 `overall_metrics.csv`、`lead_metrics.csv`、QC排除清单。两个派生组还保存增强预测。

`source_arrays.json` 指向原baseline的target/QC数组；它们不重复复制。baseline组预测也引用原缓存。元数据保持原人口定义，实际FIR使用/回退以 `postprocess_records.csv` 为准。不能把派生目录当作独立网络模型包，也不能用父分支的通用select入口误选成新checkpoint。

FIR使用/回退计数包含评分前所有pair，字段明确标 `before_input_qc` 或 `including_input_qc_exclusions`；正式r仍排除坏I记录、坏目标导联。逐pair r为诊断，最终以公共目标记录等权汇总为准。

## 4. 本地推送与服务器首次创建工作树

本地本分支提交后推送（按你已配置的远程名选择，本例为gitee）：

```powershell
cd C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-Context
git push -u gitee B5-best-context
```

服务器第一次执行：

```bash
conda activate BioFlow
cd /home/qht/huawei
git fetch gitee B5-best-context
git worktree add --detach /home/qht/huawei_B5_best_context gitee/B5-best-context
cd /home/qht/huawei_B5_best_context
git log -1 --oneline
```

工作树已经存在时，不再执行worktree add。在其中确认没有自己的未提交改动后更新：

```bash
cd /home/qht/huawei_B5_best_context
git status --short
git fetch gitee B5-best-context
git switch --detach gitee/B5-best-context
git log -1 --oneline
```

这些命令不要求在原 `/home/qht/huawei` 切分支，不处理它的未跟踪文件。

## 5. CPU功能检查：不训练、不需要模型权重

在BioFlow环境、服务器新工作树执行：

```bash
cd /home/qht/huawei_B5_best_context
python -m unittest baselines.B5_best_context.tests.test_context -v
```

这是合成功能检查，不代表ECG成绩。

## 6. 推荐：一条脚本完成基础生成与三组评估，GPU1

使用当前 `configs/b5_best.json` 固定的最佳模型包。包仍位于原仓库results，不需要复制队友V6的pt，也不需要再训练。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

run=/home/qht/huawei/results/B5-best-context/fir25_$(date +%Y%m%d_%H%M%S)

sed -i 's/\r$//' scripts/run_b5_best_context.sh
bash -n scripts/run_b5_best_context.sh

bash scripts/run_b5_best_context.sh \
  --repository-root /home/qht/huawei \
  --gpu 1 \
  --output-dir "$run"

echo "查看结果：$run/study/summary.md"
)
```

默认数据配置：

```text
/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
```

如服务器路径不同，给脚本显式传 `--config /实际/run_config.yaml`。默认模型包路径和SHA来自继承的 `configs/b5_best.json`；找不到或SHA不符会明确报错，不会悄悄加载队友旧权重。

日志放在 `${run}.log`，不放进要求为空的运行目录，避免之前tee提前创建日志导致的FileExistsError。

## 7. 可选：已有当前B5-best可靠I完整评估，直接复用

需要原评估的两任务 `prediction_uV.npy`、`target_uV.npy`、`target_quality_mask.npy`、`window_metadata.csv` 和根目录 `evaluation.json`。只有CSV报告不能复用。

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best_context

baseline=/填写已有可靠I评估的绝对目录
run=/home/qht/huawei/results/B5-best-context/fir25_reuse_$(date +%Y%m%d_%H%M%S)

bash scripts/run_b5_best_context.sh \
  --repository-root /home/qht/huawei \
  --baseline-evaluation "$baseline" \
  --output-dir "$run"
)
```

此模式不加载torch模型、不用GPU；内部仍检查来源是Heun16/K16、beta=0.75、当前可靠I评分。旧40/25目标QC-only报告会被拒绝，不能把旧范围成绩混入新组。还会核对baseline目标/QC/metadata摘要，并逐窗核对当前数据配置的Task2目标与QC，防止从另一套配对读取历史。

## 8. 单条测试预测接口

```bash
python -m baselines.B5_best_context.predict \
  --anchor-npy /path/current_observed_I_uV.npy \
  --metadata-json /path/visible_demographics.json \
  --record-key current_record_id \
  --task-id task2 \
  --input-type ecg_machine_d6 \
  --history-npy /path/legal_historical_d6_uV.npy \
  --history-mask-npy /path/legal_historical_d6_valid.npy \
  --history-record-id historical_record_id \
  --device cuda \
  --output /path/new_prediction_directory
```

当前I为 `[1,60000]` 或 `[60000]`，历史为 `[6,T]`；导联顺序I、II、III、aVR、aVL、aVF，500Hz、原始μV。历史mask同shape。缺失历史可以同时省略两个历史文件，机器会回退投影。

预测入口 `--machine-mode none/projection/fir25` 用于明确选择；默认fir25。体脂秤使用 `--input-type body_scale_d6`，当前仍返回原B5-best结果。

未来体脂秤适配器接收预测副本、可见I和HistoryRecord，返回预测与状态，不接收隐藏目标；启用后必须另做同预算/正确打乱/禁用实验。当前三组研究的“体脂秤不变”断言针对本次首轮，不是未来永久限制。

## 9. 打包报告

`run` 为本次实际运行目录。如果已退出第6节括号里的子shell，需要先重新设置为日志中显示的实际路径：

```bash
run=/home/qht/huawei/results/B5-best-context/填写本次实际目录名
tar -czf /home/qht/B5_best_context_reports.tar.gz \
  --exclude='*.npy' --exclude='*.pt' \
  -C "$run" study run.json
```

不需要打包原始ECG、完整预测或权重。先看 `study/summary.md`：C相对B是否有增益、Task1/body是否不变、机器肢体是否改善，不能只看整套相对A。

## 10. 移植来源与限制

- 来源包：`huawei-d6-context-fir25.zip`，release ID `d6-context-v6-fir25-20261017`。
- 仅复用 `baselines/context_d6_fir_postprocess.py` 为本包 `fir.py`，文件字节保持一致。
- 原模块SHA256：`bebadd12402e46d605ac6331c40b8f88fd1f0e3a054ba3995b802d758f0eb3c1`。
- 原方法seed42在旧V6参考机器组约0.622310→0.628074；它不是当前B5-best预期成绩。新主干、新QC必须实际测试。
- 队友已报告部分记录会下降，本分支不按验证真值挑患者、不增加新的holdout阈值或比例搜索。
- 不携带V6模型、患者FIR查表、原始波形、训练optimizer或人口学文件。本次创建分支与提交，不自动推送远程、不训练。
