# 方向4 N1/N2/N3小预算非线性验证

三组均以原B5-final为基线，不叠加方向2；FM/P/context固定，只修改V1--V6。

|组|修正头|
|---|---|
|N1|六个小型逐导联头，无新增低秩共享头|
|N2|2条共享曲线的可学习导联加权组合＋同样的六个小头|
|N3|3条共享曲线的可学习导联加权组合＋同样的六个小头|

共享时序主干均24通道，6个双卷积残差块，膨胀率1/2/4/8/16/29；
5Hz下感受野241点，加上输入统计覆盖约49.2秒。只使用当前I的局部均值、
RMS、MA501序列及人口学。所有归一化统计只拟合训练患者；LayerNorm只跨通道。
不把上一轮线性探针的全部滞后特征再送入CNN，以免隐含扩大时间范围。

每个导联独有头只有2个隐藏通道；三组参数量差异<1%。共享混合权重用训练集
慢误差PCA基底初始化并允许学习。全部末层零初始化，第0轮精确复现原B5-final。
主干和独有头采用相同初始化；数据顺序、seed42、AdamW lr2e-4、batch8、10轮一致。

监督为逐导联QC掩码下的归一化慢误差MSE＋0.1慢曲线PCC＋0.01总修正L2
＋0.001分支L2。按患者平衡采样权重，既有目标常量只在训练监督中去除。
最终波形为原预测加修正，修正不再额外乘beta。

每轮按正式完整120秒r选模，保留第0轮；输出共享/独有能量与交叉项，
检查是否独有支路承担全部修正。报告best及固定第10轮，不仅比较退步对照。
本轮仅候选筛查；只有超过原模型和匹配对照后，才考虑扩大训练或补seed。

```bash
python -u -m baselines.B5_final.shared_neural \
  --screen-run /root/autodl-tmp/huawei2.0/results/shared_slow_20261021_run001 \
  --output /root/autodl-tmp/huawei2.0/results/shared_neural_20261021_run001 \
  --device cuda
```

输出各组learning_curve.csv、validation/、best.pt、last.pt、final/best与final/last
的实际预测和评分，根目录summary.csv用于最终候选比较。
新权重协议B5-final-shared-private-neural-screen-v1，不可替换原base.pt。
