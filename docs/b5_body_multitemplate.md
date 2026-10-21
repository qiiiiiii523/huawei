# 方向2第二阶段：历史单模板与多模板的配对验证

固定已有 C3 五导联输出方式及 FM/P，均从 c3_joint 的 train_five/best.pt 初始化。
复用 C3 冻结训练/验证缓存；不重新生成 FM，不改变历史 B 去漂移或 O16 统计量。

## 只改变历史模板表示

- single：保留当前中位 I/II 模板，12个窗口共享，作为匹配续训对照。
- multi：使用相同最多32个有效历史心搏，只按历史 I 形态分为最多3组。
  每组保留 I/II 中位模板、数量及 I 形态离散度。
  每个当前10秒窗口只用真实可见 I 提取查询模板；按形态相似度软加权历史组，
  温度固定0.2，并用组大小及离散度作固定先验。当前提取失败按历史组先验加权。
  组间 I/II 共同加权，原27维全记录统计量不变；不把历史第k拍对齐当前第k拍。
- no_history：同模型独立续训但禁用历史，用于判断通用修正收益。

网络架构/参数量完全一致，仅提供给历史编码器的模板改变。
三组同起点、seed42、lr2e-4、10轮、batch16、原C3五肢体损失。
每轮按完整120秒平均r选模，保留各自第0轮；single第0轮必须精确复现已有训练C3。
multi初始化时表示已改变，不能声称它第0轮也精确复现single。

## 个人历史对照

对 single 和 multi 的 best 与固定第10轮，分别评估：正确历史、禁用历史、
三个固定患者级错配历史（101/202/303）。错配不保留同患者、不按结果挑捐赠者，
用接收者当前I匹配捐赠者历史模板。禁用仍有模型学习到的通用修正。
no_history 为独立训练的无历史模型。全部报告逐导联与逐配对结果，且机器/Task1/胸导联固定。

只有multi优于匹配single及固定C3，且正确历史优于禁用和全部预设错配，
收益又不是少数患者拉高，才支持“保留历史形态差异有效”。单次小幅领先需后续seed复核。
当前实验不改变既有历史复现检查容差，队友基础缓存补齐后另建目录复核。

```bash
python -u -m baselines.B5_final.body_history_study pipeline \
  --c3-run /root/autodl-tmp/huawei2.0/results/c3_joint_20261021_run001 \
  --output /root/autodl-tmp/huawei2.0/results/body_multitemplate_20261021_run001 \
  --device cuda
```

输出 features/ 保存只读历史/当前I派生特征和固定错配表；single/multi/no_history/
分别包含学习曲线、best/last权重、每轮评分、best/last历史对照。
根目录 summary.csv 和 history_controls.csv 用于总比较。所有输出目录必须全新。
新权重协议 B5-C3-history-multitemplate-v1，不可直接替换旧 body.pt。
