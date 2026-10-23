# B7服务器数据准备

服务器已知路径：代码`/home/qht/huawei_B7`，数据`/home/qht/huawei/data`，旧元数据`/home/qht/huawei/metadata`，环境`BioFlow`。下面只准备数据，不启动训练。

## 输入口径

原B5数据集的逐点监督anchor是同步d12目标记录里的I；手表/d6是独立采集的context，不与d12逐点同步。B7取消context分支以后，本入口仍使用同步d12的I。输出index明确标记`anchor_source=same_record_target_I`和`original_context_used=false`，不把手表或d6信号伪装成同步输入，也不声称已完成这些设备真实I的部署评估。

沿用subject_split、pair_manifest、raw_record_manifest与device_interpretation_qc；不重划患者，不改QC。保留原配对分组与重复目标ID，同一d12可能出现在多个组里，组名表示原配对来源，不表示B7使用了该设备的context。I的可靠性取目标QC的I位。转换入口未映射人口学表，华为缓存使用未知人口学编码；如实验需要人口学，须另提供经过核对的B7 metadata。

旧record-context-v3缓存的target并非一概不能用：若能证明同一目标12个10秒块连续、未逐窗变换且能恢复原单位，可恢复长记录。但context与target是独立索引；本次直接读取原始d12，避免误拼或混用。

## 执行顺序

在取得含`baselines/B7/huawei_index.py`的最新B7代码后，在服务器执行。输出目录必须尚不存在；失败产生的部分缓存不要当作成功缓存。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
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

第2步和第3～4步可以在第1步成功后并行；公开P训练也只依赖第1步。后续训练依赖图仍为P结束后启动A与Q，Q结束后启动B。该文档没有实现训练调度器；不得把已有best文件出现当成前一训练阶段已完成。
