# 新数据体脂秤 A/B 去漂移消融

更新：2026-10-11。主方案保持 A；B 是可选对照，不是所有信号的强制清理。本次生成派生数据并接通公共接口，不训练、不拟合模型 scale。

## 1. 是否应该做

可以做：只改变可见跨时间体脂秤 context，在同一患者划分、QC、同步 I、原始 target 和评分下比较，实验含义清楚。是否改善模型得分要训练后判断；不能将滤波后的波形视作无损真值。

| 项目 | A | B |
|---|---|---|
| 体脂秤 context | 原始完整记录中心化＋固定 scale | 整记录去漂移，滤后记录中心化＋对应固定 scale |
| 心电图机 d6 context | 原始记录中心化＋固定 scale | 相同 |
| 同步 I／target | 原始电压，只缩放 | 相同 |
| 划分／QC／配对／120 秒 target／评分 | canonical 主缓存 | 相同 |

不能滤 target 来改善分数，不能把 context 漂移补加到 target，也不强行同步跨时间波形。

## 2. 固定 B 协议

`body-context-b-hp0p2-order2-sosfiltfilt-valid-runs-v1`：500 Hz、0.2 Hz 高通，二阶 Butterworth 设计，SOS 前后向滤波。二阶指单向设计；前后向组合不等于同一个单向频响。

- 在完整原始物理记录上滤波，再切窗，不每10秒重新滤波。
- 有缺口／NaN 时每导联连续有效段分别处理，不插值、不跨缺口传播。
- odd padding，`padlen=min(7500,段长−2)`；≤2点段保持原样并审计，mask与A一致。
- 滤后从完整有效点统计每导联median；尾窗／缺口padding不参与。raw填该median，模型空间为0。
- 是离线完整记录处理，不声明流式／实时支持。

构建和测试共同调用 `ecg12gen/body_scale_detrend.py`。API依据 [SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)、[sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)。参数沿用项目旧B对照，不声称是官方或最优滤波参数。

## 3. 新缓存与旧备份

新目录：`C:\Users\Ashley\Desktop\HW\task2_body_scale_ablation`。旧目录完整备份为 `C:\Users\Ashley\Desktop\HW\task2_body_scale_ablation_legacy_20261011`。

B是body-only overlay，不复制、不改canonical target、anchor或machine context：

```text
body_scale_train_context_B.npy
body_scale_validation_context_B.npy
body_scale_train_context_valid_mask_B.npy
body_scale_validation_context_valid_mask_B.npy
body_scale_context_window_metadata_B.csv
body_scale_ablation_manifest.json
README.md
```

metadata有滤后median、独立context起点、有效长度、canonical_array_index；manifest记录参数、源文件、新数组和canonical/QC/划分指纹。Dataset拒绝旧B协议或已改变的canonical metadata。

全部134条新版body context生成B，训练仍按target/I QC：训练109配对中107可监督／1284 target窗；验证25配对／300 target窗。context窗数按自身实际长度决定，不能与target窗数混算。

本次实际生成：训练1471个独立context窗、验证339个，包含保留的尾窗。没有内部缺口、≤2点短段；主缓存和原始body文件的指纹复核一致，A/B真实检查与合成检查已通过。

默认仍A，主缓存仍 `task2_record_context_v3`。strict索引不用重建，因为target数组和位置未变。

## 4. 读取和测试

```python
from ecg12gen.dataset import JointAnchorDataset
a = JointAnchorDataset("configs/common.yaml", "task2", "train")
b = JointAnchorDataset("configs/common.yaml", "task2", "train",
                       body_scale_variant="B_detrend_0p2Hz_then_window")
```

b同时保留两设备，只替换body context。`BodyScaleVariantDataset`同名参数可读取body-only子集；正式Task2评分必须再收集machine组预测。

```python
inputs = prepare_record_context_inference(
    visible_anchor_uV, visible_raw_body_context_uV, "body_scale_d6", preprocessor,
    context_valid_mask=visible_valid_mask,
    body_scale_variant="B_detrend_0p2Hz_then_window",
)
```

传原始完整可见context，不要预滤后让helper重复滤波。机器d6保持A，缺失context仍anchor-only。输出只乘d12 scale，不补baseline。

B的body context scale只由B训练有效信号准备并冻结；d12／anchor／machine scale与对应A/P0保持兼容。checkpoint必须记录variant/filter协议与scale。目前通用preprocessor scale文件不自动强制区分A/B，模型分支必须检查实验元数据，不能给A checkpoint随意换B scale。

主loss和evaluator无需改：II–V6逐记录r，两设备1:1；各设备V1–V6 RMSE≤70得10，否则10/(RMSE/70)，分别换算后平均。

## 5. 检查和后续比较

`python scripts/check_body_scale_variants.py`：合成频率响应、缺口隔离、短段与推理一致性；真实A/B target、anchor、QC、mask、尾窗、machine context一致；原始文件／主缓存哈希一致；恒等评分一致。

重新构建：`scripts/build_body_scale_record_ablation.py --output-dir <新目录>`，拒绝覆盖现有目录。主缓存或QC/划分变化后必须重建B。

A/B训练固定初始化、预算、seed、loss、划分和评分；只改变body处理及对应冻结输入scale。先比较body分项r/RMSE/加分，再比较Task2设备等权总分。需要训练或明确的微调实验，不从生成缓存推出模型效果更好。
