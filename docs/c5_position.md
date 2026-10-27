# C5：C0的10→10华为适配＋原记录位置

## 要回答的问题

C5与C0比较：同样10秒输入、生成完整10秒时，额外提供“这段来自120秒记录第几秒”是否有益？不是把窗口缩短的收益全部算到位置上，也不把每段r高当成整段趋势已恢复。

- 对照为 `B7-C-round1` 的C0，不是C1的PCC0.3或C2～C4的附加损失方案。
- 复用实际C0的公开P最佳EMA权重，不重做P；优化器、EMA按新A阶段初始化。
- 唯一结构变量是位置投影；C0和C5主干均M7，输出II、V1～V6共7条速度，I已知，派生肢体公式不变。500Hz、Heun16、K16不变。
- C5和C0的A配方完全一致：batch32、lr1e-5、seed42、EMA0.999，最多4000步；0、1000、2000、3000、4000步验证；每250步保存checkpoint；早停patience3、min_update1000、min_delta1e-4。
- PCC11等权系数0.1，FM7/MAE7的II权重5，无额外低频、肢体波形或投影损失。

## 位置网络与数据传递

`p = original_sample_index / 30000 - 1`；输入 `[p,p²,p³]`，经3→64的1×1卷积投影后加入stem。新增256参数，总14,503,071。新增投影零初始化且不改变主干或后续噪声的初始化RNG。它是记录绝对位置，独立于FM时间t。

复用原Huawei10视图索引，不写新波形。训练用 `window_start + 裁窗内索引`；验证也从每条视图的window_start取得原位置。同一批第0～10秒与第110～120秒的切片具有不同坐标。推理短窗必须传 `metadata.record_offset`，单位为500Hz采样点、非重叠10秒起点，例如55000；缺失时报错。完整120秒输入默认从0开始，内部每10秒条件连续。不能把位置当成噪声，也不能每个视图都从0重新计时。

## 训练与评估入口

```bash
python -u -m baselines.B7.c5 \
  --control-root /home/qht/train_scheduler/runs/20261026-234809-7c690a2d/experiment \
  --manifest /home/qht/huawei/data_cache/B7_C_round1/huawei10/manifest.json \
  --full-manifest /home/qht/train_scheduler/C_full120_20261027/full120_manifest.json \
  --output results/C5_seed42 --device cuda
```

`--check-only`只核验；重复相同命令可从自己的A/last.pt恢复。配置、代码、数据哈希变化会拒绝复用输出。禁止将C0或旧A-Pos的目录作为C5恢复目录。入口不训练P/Q/B。

入口先检查实际C0规格、P最佳checkpoint、共享10秒索引及缓存哈希，保证不是另换一个P或数据。原C0–C4工作树、检查点和缓存不改。

完成A后自动评估：

1. 原C系列的非重叠10秒窗口验证r，用于同口径选best。
2. 将完整120秒按10秒输入生成后拼接，在60000点上算一次r；对窗口指标选出的best及最终EMA分别评估，规则/精度与C0–C4已有full120核查一致。不用完整120秒分数重新挑权重。

完整评估使用源记录的原QC；短窗评估使用既有view QC，两者分母不同。两种推理的keyed noise记录键也按既有各自协议保留，不把分数差全部解释成DC。正式结论需分别对照C0的对应指标，尤其关注V1/V3/V6和完整120秒收益。raw与EMA不得混比。这里不保证能达到外部全记录慢分量校正器约0.713的结果。

产物：specification.json、A/config.json、A/history.jsonl、A/validation_*.json、A/checkpoint_*.pt、A/best_r.pt、A/last.pt、full120_best_segment_selected.json、full120_final.json、completed.json。
