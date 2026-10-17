# 历史实验：原FM＋原S50-F的β选择

这是升级control100＋SlowPCC之前的历史记录。**当前默认已经变更**，见`docs/b5_best.md`；下表不代表新权重的参考成绩。

旧组合缺失11导联为：FM完整预测−501点全记录移动平均＋β×原S50-F预测慢曲线。

|推理seed|β|Task1 r|Task2 r|平均r|
|---:|---:|---:|---:|---:|
|42|0.5|0.691914|0.685830|0.688872|
|42|0.75|0.698164|0.691673|0.694918|
|42|1|0.695855|0.689403|0.692629|
|43|0.75|0.698445|0.691056|0.694750|
|43|1|0.696160|0.688680|0.692420|

旧权重上0.75优于已测0.5、1；只是同批患者上的噪声复核。RMSE不选模。

复现旧模型包须明确选择旧配置：

```bash
python -m baselines.B5_best.bundle \
  --profile configs/b5_best_legacy_beta075.json \
  --output /home/qht/huawei/results/B5-best/legacy_beta075_copy/model.pt
```

该旧导出器需要原FM、原S50-F与prepared.json；新版默认只复制已验证的完整组合包，不读取这几份训练源文件。

旧v1包没有slow_beta，仍按β=1；v2包必须显式保存合法slow_beta。显式checkpoint始终使用该包自己的参数，更新默认配置不会改变旧包输出。

历史diagnose仍分析旧原FM/S50-F数组，自动读取旧配置及reference_qc_r_beta1。新best的参考r不能用于核验这些历史数组。
