# C5 + 一秒局部均值修正网络

此分支是冻结原C5、外挂31,687参数修正头的方案，不是修改FM loss的实验。基础C5仍用EMA、原500Hz10→10/Heun16/K16；拼成真实120秒后，修正头读取同步I、七条基础预测与绝对记录位置，输出每秒II、V1–V6七个修正量，重复500点加回。其余肢体导联由固定I及修正II恢复，I保持原始输入。

## 结构与训练

原始电压按固定训练尺度归一化，块平均到50Hz。9→32 Conv5/GELU→32 Conv5/GELU→每秒池化→四个双Conv3残差块（dilation1/2/4/8、dropout0.1）→零初始化七通道输出。时间上下文总计约61秒，离线非因果。

修正头监督为真值均值减基础预测均值；QC/物理目标/设备组/导联加权Huber＋0.1完整120秒PCC。原实验359条训练配对/156位受试者、90条验证配对/39位受试者，受试者隔离；AdamW1e-3、weight_decay1e-4、batch16、100epochs、seed42。每5epochs验证，以完整120秒r选择best，包括零修正起点候选。基础C5完全冻结。评分通过段内二阶矩精确计算整段相关性，不是逐秒相关性平均。

分支包含代码与合成测试；私有训练数据、原始预测缓存、病人级清单、基础权重和修正头权重均需另行提供。不能把真值均值替换诊断当作可部署成绩。修正头表现须以实际完整120秒验证判断；单seed验证选点不是独立测试确认。每秒常数修正可能产生边界跳变。

## 测试

在仓库根目录执行：

```bash
python experiments/c5_mean_correction/test_mean_model.py
```

依赖沿用仓库PyTorch、NumPy。测试检查完整评分矩、目标不进入特征、零输出、肢体公式、可学习性和独立推理入口。

## 推理

```bash
python experiments/c5_mean_correction/predict.py \
  --base-checkpoint /path/to/C5/best_r.pt \
  --corrector-checkpoint /path/to/corrector/best.pt \
  --anchor /path/to/I_120s_raw_uV.npy \
  --record-key stable-record-key \
  --output /path/to/new_prediction.npy --device cuda
```

输入I形状[1,60000]（或[60000]），单位µV。可用`--metadata`提供numeric、sex、field_mask、age_topcoded字段。record-key决定基础模型固定噪声，复现时须保持一致。严格核对修正头训练时的基础权重SHA256、采样配置和尺度；需要原C5的best_r.pt文件，不能换同架构的其他检查点。

输出[12,60000]，导联顺序沿用仓库；同时保存[7,120]修正量及来源JSON。推理不需要真实目标或目标QC。

## 复现原可学习性实验

`run_experiment.py`保留原有数据身份校验，训练队列脚本不在本分支中自动提交。路径可通过环境变量配置：`C5_PROJECT`（默认仓库根）、`C5_QUEUE`、`C5_BASE_CHECKPOINT`、`C5_VIEWS`、`C5_VALIDATION_CACHE`、`C5_FULL_VALIDATION`。默认服务器路径仅用于复现既有实验，不代表仓库包含相应文件。

```bash
python experiments/c5_mean_correction/run_experiment.py prepare
python experiments/c5_mean_correction/run_experiment.py cache --shard 0 --shards 3 --output /new/cache0
python experiments/c5_mean_correction/run_experiment.py cache --shard 1 --shards 3 --output /new/cache1
python experiments/c5_mean_correction/run_experiment.py cache --shard 2 --shards 3 --output /new/cache2
python experiments/c5_mean_correction/run_experiment.py train \
  --shard-dirs /new/cache0 /new/cache1 /new/cache2 --output /new/corrector
```

prepare写入本地plan.json（已忽略），运行前须备好同协议基础验证缓存。训练检查数据、代码和缓存身份；若改变协议，请新建输出及计划，不能混用旧缓存。best.pt与last.pt采用`model`、`epoch`、`specification`结构，可直接交给上面的推理入口。
