> **文档状态：** 本文保留原始数据整理与审计规则。当前主分支的数据窗口、D0/D1/V0 和训练协议以仓库根目录的 `README.md`、`configs/`、`metadata/` 为准。

# ECG 比赛数据整理项目

本项目用于 ECG 比赛的数据索引、配对、质量检查和受试者级划分。原始数据目录 `Data` 只读，禁止修改、移动、重命名或删除。v0.1 的 D0 直接消费团队已生成的最终 10 秒窗口；本仓库不重新切分原始记录，也不包含模型训练实现。

## 一、项目目录和文件

- `metadata/raw_record_manifest.csv`：原始记录索引。保存 `record_id`、原始 `externalid`/`groupid`、统一 `subject_id`、路径、时间、采样率、单位、解析状态和 QC 状态。
- `metadata/pair_manifest_task1.csv`：任务一配对，手表单导联 ECG -> d12；包含配对状态、目标复用次数和训练策略。
- `metadata/pair_manifest_task2.csv`：任务二配对，心电图机 d6 -> d12、体脂秤 d6 -> d12；包含体脂秤导联顺序和通道映射。
- `metadata/subject_split.csv`：已确认的正式划分，是后续所有任务唯一的 train/validation 依据。
- `metadata/id_mapping_review.csv`：不能自动可靠统一的 ID，必须人工确认。
- `metadata/field_mapping.md`：字段、配对、采样率和单位的详细解释。
- `configs/data_spec.yaml`：统一数据规范，包括 ECG 内部单位、采样率、导联顺序和窗口参数。
- `configs/quality_rules.yaml`：质量状态和基础 QC 处理规则。
- `reports/inventory_report.md`：原始文件清点报告。
- `reports/qc_report.md`：基础质量检查报告。
- `reports/reject_records.csv`、`reports/review_records.csv`：QC reject/review 记录。
- `reports/duplicate_pair_candidates.csv`：重复目标和跨设备目标复用审计。
- `reports/phase1_audit_report.md`：第一阶段审计及划分统计。
- `reports/final_split_report.md`：正式 subject split 的一致性和覆盖统计。

## 二、当前数据概况

清点报告记录 632 个文件；manifest 有 728 条逻辑记录，因为一个 ZIP/XML/CSV 文件可能包含多条记录。当前 manifest 质量状态为：usable 616、review 110、reject 2。

任务一共有 117 条 paired、4 条 review、9 条 unmatched；任务二有 100 条 machine d6 -> d12 和 34 条 body-scale d6 -> d12，均为 paired。可用于监督训练的候选数量为：任务一 104、任务二 machine d6 100、任务二 body-scale d6 34。

## 三、后续任务应读取的文件

### 任务一：手表 ECG -> d12

必须读取：

1. `metadata/raw_record_manifest.csv`：定位记录、读取路径、采样率、单位和 QC 状态；
2. `metadata/pair_manifest_task1.csv`：确定输入记录、目标记录和 `pair_status`；
3. `metadata/subject_split.csv`：按 `subject_id` 选择 train 或 validation；
4. `configs/data_spec.yaml`：使用 ECG 内部单位、500 Hz、十二导联/六导联顺序和窗口参数；
5. `configs/quality_rules.yaml`：执行统一质量筛选；必要时参考 `metadata/id_mapping_review.csv` 排除未确认关系。

只有 `pair_status=paired` 且输入、目标质量均为 `usable` 的记录才是监督候选。多个手表 ECG 共用同一个 d12 是预期的重复目标使用，不是错误配对；后续采样需要按 subject 或 target 均衡，不得逐点平均 ECG 波形。

### 任务二：d6 -> d12

必须读取：

1. `metadata/raw_record_manifest.csv`；
2. `metadata/pair_manifest_task2.csv`；
3. `metadata/subject_split.csv`；
4. `configs/data_spec.yaml` 和 `configs/quality_rules.yaml`；
5. `metadata/field_mapping.md`，尤其是体脂秤六导联映射。

machine d6 和 body-scale d6 是不同的 `input_type`。同一个 d12 被两种设备复用是预期的跨来源目标复用，不能标记为重复错误。体脂秤通道 1、2、9、10、11、12 固定对应 I、II、III、aVR、aVL、aVF，原始 500 Hz/mV，后续预处理再执行 `mV × 1000 -> μV`。

## 四、当前固定划分

`subject_split.csv` 是 `subject_split_proposed.csv` 的原样复制，使用固定随机种子 42。共有 110 个受试者：train 88 个、validation 22 个；同一 `subject_id` 的所有设备和记录必须位于同一集合。train/validation 的可用配对分别为：

| 集合 | 手表 ECG -> d12 | machine d6 -> d12 | body-scale d6 -> d12 |
|---|---:|---:|---:|
| train | 83 | 80 | 29 |
| validation | 21 | 20 | 5 |

不得按文件、记录或窗口重新随机划分，也不得自行生成另一份 split。

## 五、可以修改的参数

以下参数只能由项目负责人或经组员确认后修改，并必须在提交记录中说明原因：

- `configs/data_spec.yaml` 中的窗口长度和步长；
- 后续确定比赛要求后，ECG 模型内部采样率、内部单位或最终输出采样率（当前最终输出仍为 `TBD`）；
- 后续实验是否启用 PPG 或 acceleration，以及对应的模型输入配置；
- `configs/quality_rules.yaml` 中的时长阈值和异常处理阈值，但必须保留 `usable/review/reject` 三种状态。

参数修改只能影响后续处理，不能回写原始数据，也不能静默改变已有 QC 或配对结论。

## 六、组员不得自行修改的规则

- 不得修改、删除或覆盖原始数据；
- 不得重新生成或改写已有 `record_id`、原始 `externalid`、`groupid` 或 `subject_id`；
- 不得修改 `subject_split.csv` 的 train/validation 映射；如需变更，必须重新评审并形成新的确认记录；
- 不得修改已有 `pair_status`、`pair_relation`、`training_policy` 或 `target_reuse_type`；
- 不得猜测不确定的跨设备 ID 或 d6/d12 配对；
- 不得改变体脂秤六导联映射和固定导联顺序；
- 不得把 d6 中 V1-V6 全零直接判为文件异常；
- 不得因为单个 ECG 波峰幅值较大就删除该点；
- 不得把多个 ECG 波形逐点平均来消除目标复用；
- 在负责人确认前，不得创建新的 subject split、切最终窗口、导出最终训练数据或训练模型；
- 比赛最终输出采样率未知时必须保持 `TBD`，不得自行猜测。

## 七、执行原则

任何后续脚本都应先读取 `subject_split.csv`，再读取对应任务配对表和 manifest；不能绕过 subject-level 划分直接按文件或窗口拆分。发现冲突、缺失映射或新的异常时，保留原记录并标记 review，提交人工确认。
