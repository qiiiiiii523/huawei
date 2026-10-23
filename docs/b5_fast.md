# B5-fast：固定原最佳 FM 快分量，预留新慢分量接口

2026-10-24。Git分支 `B5-fast`，源 `B5-1022@69bc40f`。本次不训练FM、不改成七通道、不实现任何新慢网络。继承旧仓库的历史代码仅为来源记录，本分支的新入口统一为 `fast_main.py` / `baselines.B5_fast`；不要使用历史 `competition_main.py`、慢CNN或G120脚本。

## 固定什么

FM使用B5-1022正式组合里的 `FM_P_best45` EMA，原11个缺失导联生成网络保持原样，参数全部冻结、eval模式；同步I和人口学条件与旧FM一致，I不加生成噪声。

来源包：`FM_P_best45__P.pt`（本地副本为原 `B5-1022-EXE-assets/models/base.pt`）。包SHA：`990f5d95312dba0ab3e4bde416ee15d3b464b902fc63601aab1f852d9881f6f8`。

原始FM checkpoint SHA：`74a5e401233c1593b5d7b22da48510a5539e0c65befec5e3daa270809d7e2603`；FM tensor SHA：`990e92e30f5255046c6aa73ab55f1ef4e22fc814a05f4fc92c14457bd6d35651`。提取后参数逐张量严格加载，不迁移、不重新初始化。

本轮用户明确选择Heun16＋8对正负噪声(K16)、scale1、seed42。第i对正噪声沿用旧 `B5-v1|42|record_key:start_sample|i` 种子规则，成员顺序为z0,-z0,...,z7,-z7。原正式包是独立K16；因此新默认快分量不宣称与独立K16的旧预测逐点相同，也不套用旧组合分数。

FM仍按12个不重叠10秒窗口生成并平均，先恢复μV、拼好完整120秒，再进行原501点滑动平均、端点重复padding：

`fast_missing11 = FM_missing11 - MA501(FM_missing11)`。

这是原B5的工程快残差，不是新训练出来的高频专用FM，不是理想截止频带。不能逐10秒分别平滑，否则接缝处结果会改变。fast用float64保留减法，最终组合才转float32，与原combine舍入顺序一致。

运行不加载P慢CNN、G120、body/C3、机器历史FIR，也不使用目标标签。原I输出头仍保留在原网络中以保证参数一致，但组件只导出缺失11导联；最终组织12导联诊断文件时直接使用可见I。

## 独立模型包

标准环境安装 `pip install -r requirements-fast.txt`，然后：

```bash
python -m baselines.B5_fast.package \
  --source /实际位置/FM_P_best45__P.pt \
  --output models/b5_fast.pt
```

本机已从已核验本地包导出 `models/b5_fast.pt` 和 `models/b5_fast.json`。模型二进制按仓库规则不进Git；随代码拷贝这一份独立包即可推理，运行不再需要原混合包。`models/b5_fast.json`包含包SHA和完整协议，实际加载会核对FM参数SHA、架构、scale与冻结协议。

独立包仅包含FM参数、scale、schema及来源；不含慢CNN、优化器、原始ECG或训练缓存。Git内 `configs/b5_fast.json` 钉住模型、采样、分解和接口；可用--checkpoint搬迁包，但不得默默改参数或采样。如将来要试另一种采样，应新建有明确身份的实验协议。

## 单记录推理

```bash
python fast_main.py \
  --checkpoint models/b5_fast.pt \
  --anchor-npy /输入/current_I_uV.npy \
  --metadata-json /输入/demographics.json \
  --record-key 稳定记录ID \
  --output /新的结果目录 \
  --device cuda --batch-size 1
```

I为原电压μV、500Hz、完整120秒，形状[1,60000]或[60000]。人口学JSON使用原age/sex/height/weight字段，可省略缺失字段；record-key和采样协议固定才可复现。

输出：

- `fast_missing11_uV.npy`：[11,60000]，float64，II、III、aVR、aVL、aVF、V1～V6；这是未来慢网络的固定搭档。
- `observed_I_uV.npy`：[1,60000]，原可见I。
- `fast_only_d12_uV.npy`：[12,60000]，可见I+快残差，供检查；缺少慢分量，不能当作已完成的12导联完整重建或直接替代正式提交。
- `inference.json`：record-key、单位/坐标、数组SHA、权重SHA、噪声配对、分解方式、零训练更新等。

输出目录必须全新，避免覆盖旧记录。GPU通过batch-size调显存；噪声与batch顺序无关，计算舍入在不同设备/批次可能有微小差异，比较时固定设备和batch设置。

## 未来慢分量接口（只定义接口，没有慢模型）

新慢模型将来提供**最终要相加的物理电压慢曲线**：11条缺失导联、500Hz、120秒、[11,60000]、μV、相同record-key和起点0。它可在内部采用FM+Transformer、7通道或其他结构，但接口前必须转换到这11条导联；不把新结构强加给固定快分量。

```python
from baselines.B5_fast import COMPONENT_PROTOCOL
from baselines.B5_fast.components import WaveComponent, compose
from baselines.B5_fast.predict import Predictor

engine = Predictor("models/b5_fast.pt", "cuda")
fast = engine.predict(current_I_uV, demographics, record_key)
# future_slow_uV由将来单独实现的模型提供，本分支没有该模型。
slow = WaveComponent(record_key, future_slow_uV, "slow", COMPONENT_PROTOCOL)
complete = compose(current_I_uV.reshape(1, 60000), fast, slow)
```

compose只做 `missing11 = fast + slow`，系数固定1；不会偷偷使用原beta0.75、旧CNN或历史修正。新慢模型如需缩放，应在自己的明确协议内输出最终慢贡献。接口检查形状、单位约定、采样率、导联顺序、记录与起点；组件复制数组并设只读，组合不会原地改固定快残差。当前 `compose(anchor, fast)` 只产生快分量诊断。

以后训练慢网络时冻结FM；可缓存每条记录的fast、原可见I及inference身份，验证缓存的权重/采样/数组SHA后复用，避免每轮重复生成。训练未来慢网络的loss与目标本次不规定、不实现。

## 功能验证

```bash
python -m unittest discover -s baselines/B5_fast/tests -v
# 包已导出时，增加真实已选EMA的加载/冻结/篡改检查：
B5_FAST_TEST_PACKAGE=models/b5_fast.pt python -m unittest discover -s baselines/B5_fast/tests -v
```

测试覆盖原combine数值等价、跨10秒接缝平滑、正负噪声配对与旧种子、干净I与K16平均、完整12窗拼接、冻结/参数身份、外部慢输出及记录错配拒绝。合成/真实权重功能检查不表示新的验证性能。测试及真实推理自检结果记录在 `reports/b5_fast_verification.json`。

## 服务器分支使用

本轮只创建本地分支，不自动推送。之后同步代码和独立模型包：

```bash
git fetch origin B5-fast
git worktree add /实际位置/huawei_B5_fast origin/B5-fast
cd /实际位置/huawei_B5_fast
# 将本机models/b5_fast.pt拷到此处models/，或使用原base包执行上面的package提取。
```

不要从旧慢分支的脚本启动训练。本分支的B5_fast模块没有优化器或训练入口；继承的历史源代码仅作兼容参考。未来慢FM+Transformer应从此分支另建实验分支，保留本分支快分量协议不变。
