# B5：对齐最新 main 的条件 Flow Matching 基准

已接入 main `6dcb01c`（新规则新数据V2）。网络仍为64/128/256条件1D U-Net，同步I多尺度条件＋人口学FiLM；FM路径、原损失与参数布局不变。没有T/F分支，也没有异步context编码器。

训练使用新版strict去重索引；验证使用新版独立记录Dataset的全部target窗，context仅用于组织正式配对和设备来源，不输入网络。不要把这个基准称为已经完成context融合的P1模型。

最终推理默认EMA、Heun16、K16、seed42。每个正式target为真实120秒，Task2分别评分机器与体脂秤并1:1汇总，胸导联加分分别换算后再等权。

旧raw-target B5 checkpoint可用于新数据初始化或新验证集评估，尺度必须复用原checkpoint对应的文件。新数据不能直接resume旧last.pt；必须开新输出目录，重新建立验证最佳分数。新checkpoint绑定数据缓存字节、索引、配对、QC、人口学与采样/评分协议。

服务器上传目录、预检、重新评估、微调、续训、推理命令及旧新成绩的比较限制，见 [最新版main接入说明](../../docs/b5_main_v2_migration.md)。本分支仅保留 `b5_base.yaml` 和 `b5_finetune_meta.yaml`，最终采样配置为 `configs/b5_inference.yaml`。

CPU功能测试不运行optimizer更新：

```bash
python -m unittest discover -s baselines/B5/tests -v
```
