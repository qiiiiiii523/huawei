# C3 五肢体联合训练

冻结 base.pt 中 FM/P，初始化自当前 body.pt，继续训练原 BodyIIAdapter。
原基础 II、同步 I 和按原策略处理一次的 B/O16 历史仍是唯一模型输入。
网络只生成一条 II 修正，先对原五肢体预测做固定 I 的最小二乘投影，再按
`[1,1,-0.5,-0.5,1]` 传播该修正。I、胸导联、机器路线和 Task1 固定。

## 首轮配对

- five：五肢体目标的 QC 掩码 PCC 联合监督。
- II：相同 C3 输出结构，但仅用 II 目标监督，作为配对训练对照。
- 相同 body 初始化、seed42、10轮、batch16、AdamW lr=2e-4、weight_decay=.01、梯度裁剪1。
- 损失为窗口 PCC 加 0.01×归一化修正 L2；重复目标配对按倒数权重训练。
- 完整120秒官方评分、逐目标及导联QC、设备等权；按完整平均r选模，并保留第0轮。
- 每轮输出 Task1/机器/body/Task2/总r、逐导联与逐配对诊断；保存 best 和 last。
- 当前运行的基础缓存只是临时计算来源，暂缓历史严格复现检查；不更改任何历史容差。

## 运行

```bash
python -u -m baselines.B5_final.c3_joint pipeline \
  --source-run /root/autodl-tmp/huawei2.0/results/body_scope_20261021_run001 \
  --output /root/autodl-tmp/huawei2.0/results/c3_joint_20261021_run001 \
  --device cuda
```

先检查当前 C3 第0步精确一致，然后只生成体脂秤训练集基础预测。
验证集沿用已有基础缓存；历史特征缓存与训练/验证患者、目标均隔离校验。
缓存完成后依次执行 five 和 II 两组训练，全部写入独立输出目录。
缓存准备可通过 `prepare --resume` 从完整记录恢复；训练不覆盖旧目录。

`cache/cache.json` 固定数据/权重/预测源哈希。新权重协议是
`B5-C3-five-limb-continuation-v1`，不能直接当原 body.pt 交给旧的仅II部署入口。
队友缓存补齐后应另建新缓存/结果目录复核，不静默替换已有训练缓存。

本地测试：`python -m unittest discover -s tests -p test_c3_joint.py -v`。
