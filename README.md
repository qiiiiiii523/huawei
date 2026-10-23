# B5-fast：固定原最佳B5快分量

本分支从 `B5-1022@69bc40f` 创建，只提供固定快分量的新入口与未来慢分量的相加接口。

```text
当前同步I＋人口学
       ↓
原FM_P_best45 EMA，全部参数冻结
Heun16＋8对正负噪声(K16)，scale1，seed42
       ↓
12个10秒窗口生成，拼成完整120秒FM波形
       ↓
缺失11导联 - 全记录MA501(缺失11导联)
       ↓
固定快分量 [11,60000]，μV
       ↓
未来可加新慢分量，本次没有实现慢模型
```

- 原11通道FM及I条件编码保持原样，不采用B7-U7改造，不重新训练。
- 新运行入口不加载慢CNN、G120或历史context。旧源代码留作历史兼容，正式提交入口在此分支停用，避免误调用旧组合。
- 独立模型为 `models/b5_fast.pt`（本机已导出，二进制不进Git）；FM参数与原包逐张量相同。
- 快分量单独输出不代表完整重建效果，不沿用旧组合成绩。

使用方法、模型身份、未来慢接口及验证说明见 [docs/b5_fast.md](docs/b5_fast.md)。

```bash
pip install -r requirements-fast.txt
python fast_main.py --help
python fast_main.py --anchor-npy /path/I_uV.npy --record-key stable_record_id --output /path/new_fast_output --device cuda
```

服务器没有独立包时，可从同一已核验的原base包提取：

```bash
python -m baselines.B5_fast.package --source /path/FM_P_best45__P.pt --output models/b5_fast.pt
```

生成只含快分量代码和模型、没有历史慢代码的独立交付包：

```bash
python scripts/build_b5_fast_release.py --output release/B5-fast.zip
```

包内不含训练数据或慢CNN。将来从本分支另开FM+Transformer慢分量实验，保持本分支快分量协议固定。
