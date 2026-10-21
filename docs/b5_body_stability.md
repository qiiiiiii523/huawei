# 当前单模板 C3 续训的稳定性验证

固定 c3_joint/train_five/best.pt 作为三个种子的共同初始化。
复用已完成的 body_multitemplate/single seed42，新增 seed43、44。
只改变训练随机性/批次顺序，不改变单模板、C3输出、五肢体损失、数据、
lr2e-4、10轮、batch16、FM/P权重及其推理噪声seed42。

这验证的是“给定同一已训练C3起点，这次单模板续训能否重复提分”，
不是把FM/P/C3从头训练三遍，也不能据此声称全流程随机性已验证。
不挑最高seed，每个seed同时报告best和固定第10轮；全部保留第0轮。
各seed的best/last继续输出正确、禁用和三个固定错配历史对照。

对24位有效验证患者做5000次配对患者重采样。患者重复配对先在目标内平均，
按原逐导联QC与目标等权计算11导联均值，body贡献按1/4进入最终mean_r。
三seed使用同一组重采样索引；另报告seed均值的配对患者95%区间。
该区间只描述当前验证样本的条件不确定性，不纠正反复选模，也不替代独立验证。

```bash
python -u -m baselines.B5_final.body_stability \
  --history-run /root/autodl-tmp/huawei2.0/results/body_multitemplate_20261021_run001 \
  --output /root/autodl-tmp/huawei2.0/results/body_stability_20261021_run001 \
  --device cuda
```

顺序运行seed43/44，然后生成summary.csv和completed.json，记录全部seed、
best/last、增量、患者区间、均值与标准差。队友缓存复现检查仍按原计划暂缓。
