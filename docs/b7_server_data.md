# B7-T服务器准备与自动调度

服务器已知路径：代码`/home/qht/huawei_B7_T`，数据`/home/qht/huawei/data`，旧元数据`/home/qht/huawei/metadata`，环境`BioFlow`。代码目录`/home/qht/huawei_B7_T`是建议的新T工作树路径，启动前需存在并切到B7-T分支。自动调度段启动真实训练，后续手动prepare段只准备数据。

## 推荐：一次启动准备与全部训练

取得本分支B7-T代码后，不必逐条执行下方prepare。在服务器复制执行下面整段。它会启动真实训练，默认同一张GPU 0最多并行两个训练进程；单个准备进程与训练可重叠。使用独立的`auto_seed42/data`缓存，不读取其他目录的半成品。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T
mkdir -p results/B7-T/auto_seed42
tmux new-session -d -s B7_T_auto "cd /home/qht/huawei_B7_T && '$(command -v python)' -u -m baselines.B7.pipeline --data-root /home/qht/huawei/data --metadata /home/qht/huawei/metadata --ptb-root /home/qht/huawei/data/PTB-Diagnostic --output results/B7-T/auto_seed42 --gpu-p 0 --gpu-a 0 --gpu-b 0 > results/B7-T/auto_seed42/pipeline.log 2>&1"
```

启动器使用当前BioFlow的Python绝对路径，不依赖tmux旧服务的conda环境。tmux会话已存在时命令拒绝重复启动。总控还对同一输出目录加进程锁，避免多个调度器同时写同一权重。

流程：索引及公开缓存准备→P；公开缓存完成后也准备PTB和华为缓存；P与对应数据均完成后启动A/Q；Q成功后启动B，无需等A。失败仅阻止后代依赖，独立路线继续。每个训练正常返回且有正确completed.json及可加载最佳权重才启动后继，不以best文件出现作为完成信号。

```bash
# 查看总控进度；Ctrl+C仅退出查看日志
tail -f /home/qht/huawei_B7_T/results/B7-T/auto_seed42/pipeline.log

# 查看训练细节（P/A/Q/B）
tail -f /home/qht/huawei_B7_T/results/B7-T/auto_seed42/logs/P.log
```

所有产物在`results/B7-T/auto_seed42`：`data`是缓存；`P/A/Q/B`各自保存best_r.pt、best.pt、last.pt及训练历史；`logs`是逐任务日志；`pipeline_status.json`记录任务完成、失败和依赖阻塞状态。

`pipeline_status.json`的`training_results`统一汇总已成功阶段的最佳验证r、最佳/最后权重绝对路径及完成步数。训练完成判定还会核对completed、last和best的分数及身份一致性，并严格加载最佳模型；不会仅检查best文件存在。P/Q的r来自各自公开验证集，A/B来自同一华为验证口径，只有A/B用于路线效果比较。

两个GPU时将`--gpu-b 0`改为`--gpu-b 1`；同卡并行显存不足时加`--max-train 1`。GPU编号须是分配给你的GPU；本脚本不会自动估算显存。仅查看依赖与命令、不创建缓存或训练，可在Python pipeline命令末尾加`--dry-run`。

断线不需重启；总控结束或服务器重启后，重新执行同一段启动命令即可检查已完成任务并恢复有last.pt的中断训练。代码、配置、元数据与原始数据应保持不变；完成训练还会检查prepared数据hash。准备数据中途失败的半成品缓存默认报错并保留；训练尚未开始时可按下节显式核验恢复，否则检查日志并使用新output。没有last.pt的训练目录仍报错保留，不自动删除。训练恢复从最近保存的验证checkpoint继续，不保证从中断的那一步继续。不是无条件自动重试或服务器开机自启动。

## 复用B7数据与模型区别

数据协议没有改变，现有B7 manifest及其冻结scale可以通过绝对路径交给单阶段train入口复用。自动pipeline默认在自身output/data里新建专用缓存；它不会自动导入其他run的缓存或权重。T不接受U7 checkpoint，必须重新从随机T训练P，后续为T的A/Q/B。

## PTB-Diagnostic根目录副本处理

B7-T的新运行已直接修复PTB自动索引，无需另外运行B7的补救脚本。只收集数据根目录下的`patient数字/记录.hea`。例如`patient279/s0533_re.hea`保留，根目录`s0533_re.hea`跳过；不把根目录名称当患者，也不删除原始文件。其他没有患者归属的根目录头文件同样排除。

记录与target ID改为`ptbdb:patient279/s0533_re`，subject仍为`ptbdb:patient279`，划分仍使用原来的患者hash。若出现backup/patient279或patient279/子目录中的头文件，报错要求核对，不任意选副本。只有平铺头文件、没有患者目录的数据也会报错，不编造患者ID。

正常pipeline启动已自动使用这项规则，不需要额外参数。`prepare_ptb.log`中的`[ptbdb-index]`及PTB manifest的`public_index_audit`会记录头文件总数、保留数和根目录排除名单。不要把用户报告的550份扫描数直接当作有效样本数；以实际患者目录索引结果为准。

## 数据准备中断后的核验恢复

已同步B7@0113fbc：Chapman的JS23074等头文件可能把第一条信号描述粘在记录首行。读取器在内存中拆开，仍校验12导联完整性、初值和checksum，不改原始文件。B7-T已对本地真实JS23074读取通过。

如果仅数据准备失败、该output下没有P/A/Q/B训练目录，且准备缓存尚未生成完整manifest，更新修复后可保留半成品，在原pipeline命令加`--retry-preparation`。例如在B7-T工作树、BioFlow环境执行：

```bash
python -u -m baselines.B7.pipeline \
  --data-root /home/qht/huawei/data \
  --metadata /home/qht/huawei/metadata \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --output results/B7-T/auto_seed42 \
  --gpu-p 0 --gpu-a 0 --gpu-b 0 \
  --retry-preparation
```

仍可把这条命令放在上述tmux启动命令中。不要同时启动两个相同output的调度器。

此选项向三个prepare任务传递`--resume`：每条原始记录重新解码，全部NPZ数组的名称、形状和内容须一致；一致的缓存保留字节及mtime，缺少的记录继续写，不一致就报错。每250条报告进度；核验仍需要时间。必要时备份旧pipeline_spec后才采用修复版本；原始文件、元数据、患者划分与QC必须保持原样。

它不是跨版本训练恢复开关：只要P/A/Q/B任一训练目录存在，就拒绝使用；已生成完整manifest的缓存也不能经该选项跨版本直接采信。开始训练后的普通恢复须去掉该选项、保持原训练代码/配置一致；代码变更时不能绕过训练身份校验。新运行直接使用正常启动命令即可，不需要加该选项。

## 输入口径

原B5数据集的逐点监督anchor是同步d12目标记录里的I；手表/d6是独立采集的context，不与d12逐点同步。B7取消context分支以后，本入口仍使用同步d12的I。输出index明确标记`anchor_source=same_record_target_I`和`original_context_used=false`，不把手表或d6信号伪装成同步输入，也不声称已完成这些设备真实I的部署评估。

沿用subject_split、pair_manifest、raw_record_manifest与device_interpretation_qc；不重划患者，不改QC。保留原配对分组与重复目标ID，同一d12可能出现在多个组里，组名表示原配对来源，不表示B7使用了该设备的context。I的可靠性取目标QC的I位。转换入口未映射人口学表，华为缓存使用未知人口学编码；如实验需要人口学，须另提供经过核对的B7 metadata。

旧record-context-v3缓存的target并非一概不能用：若能证明同一目标12个10秒块连续、未逐窗变换且能恢复原单位，可恢复长记录。但context与target是独立索引；本次直接读取原始d12，避免误拼或混用。

## 执行顺序

在取得含`baselines/B7/huawei_index.py`的最新B7代码后，在服务器执行。输出目录必须尚不存在；失败产生的部分缓存不要当作成功缓存。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T
python -m unittest discover -s baselines/B7/tests -v

# 1. 公开短窗缓存；这一步只准备数据、拟合训练集scale，不训练。
python -m baselines.B7.prepare \
  --public-source public10 \
  --root /home/qht/huawei/data \
  --output results/B7/data_public10

# 2. PTBDB目录实际叫PTB-Diagnostic。读取器按WFDB头选择标准12导联。
python -m baselines.B7.prepare \
  --public-source ptbdb \
  --root /home/qht/huawei/data/PTB-Diagnostic \
  --scale-manifest results/B7/data_public10/manifest.json \
  --output results/B7/data_ptb40

# 3. 转换既有华为元数据；仅输出索引和audit，不改原始XML。
python -m baselines.B7.huawei_index \
  --metadata /home/qht/huawei/metadata \
  --data-root /home/qht/huawei/data \
  --anchor-source same-record-target-I \
  --output results/B7/huawei_index.json

# 4. 从连续XML重建500Hz完整120秒记录，并复用公开训练scale。
python -m baselines.B7.prepare \
  --index results/B7/huawei_index.json \
  --scale-manifest results/B7/data_public10/manifest.json \
  --output results/B7/data_huawei
```

默认d12目录为`DATA_ROOT/心电图机d12`；目录名称不同时给索引命令加`--d12-root /实际路径`。同名文件仅在指定d12目录中匹配，发现歧义就报错。不存在的文件、跨患者/划分、无效QC、验证组缺失或缺少可评估导联也会报错。

四步成功后，三个训练入口分别是：

- `results/B7/data_public10/manifest.json`：PTB-XL＋Chapman真实10秒。
- `results/B7/data_ptb40/manifest.json`：PTBDB真实长记录，40秒训练只选长度足够的记录，不把短记录拼接变长。
- `results/B7/data_huawei/manifest.json`：完整120秒记录，训练取40秒窗口、验证中央核心拼接120秒。

公开P训练只依赖第1步。后续训练依赖图仍为P结束后启动A与Q，Q结束后启动B。上面的pipeline入口已经实现这些准备与训练依赖，不需要人工等待。
