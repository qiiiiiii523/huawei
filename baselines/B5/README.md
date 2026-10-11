# B5：对齐最新 main 的条件 Flow Matching 基准

已接入 main `6dcb01c`（新规则新数据V2）。网络仍为64/128/256条件1D U-Net，同步I多尺度条件＋人口学FiLM；FM路径、原损失与参数布局不变。没有T/F分支，也没有异步context编码器。

训练使用新版strict去重索引；验证使用新版独立记录Dataset的全部target窗，context仅用于组织正式配对和设备来源，不输入网络。不要把这个基准称为已经完成context融合的P1模型。

最终推理默认EMA、Heun16、K16、seed42。每个正式target为真实120秒，Task2分别评分机器与体脂秤并1:1汇总，胸导联加分分别换算后再等权。

旧raw-target B5 checkpoint可用于新数据初始化或新验证集评估，尺度必须复用原checkpoint对应的文件。新数据不能直接resume旧last.pt；必须开新输出目录，重新建立验证最佳分数。新checkpoint绑定数据缓存字节、索引、配对、QC、人口学与采样/评分协议。

服务器上传目录、预检、重新评估、微调、续训、推理命令及旧新成绩的比较限制，见 [最新版main接入说明](../../docs/b5_main_v2_migration.md)。基础配置为 `b5_base.yaml` 和 `b5_finetune_meta.yaml`；双实验另提供两份入口配置，最终采样配置为 `configs/b5_inference.yaml`。

CPU功能测试不运行optimizer更新：

```bash
python -m unittest discover -s baselines/B5/tests -v
```

## 新版华为双实验

当前支持同配置的华为从头训练与公开权重初始化微调。原网络、损失、100轮、每5轮验证Heun16/K16，最终同样Heun16/K16；两组只改变初始化来源和输出目录。入口与命令见 [双实验说明](../../docs/b5_main_v2_public_vs_scratch.md)。

## 评分与权重保存更新

主分为50×(0.5r1＋0.5r2)，加设备等权RMSE加分，满分60。best.pt按周期验证总分选；另保存best_r.pt（两任务相关性均值）、best_bonus.pt（等权加分）、last.pt（续训）。r、各设备RMSE和全部分数仍保留在验证日志及validation_metrics.csv。

周期验证与最终生成均为Heun16/K16。新运行使用score60_k16输出目录，旧权重仍可评估或初始化，但旧选模状态不能resume。具体操作见[评分及四份权重说明](../../docs/b5_score60_checkpoint_policy.md)。

本次K16选模与main适配核对、局限及后续实验顺序见[验证采样与适配说明](../../docs/b5_k16_validation_alignment.md)。
