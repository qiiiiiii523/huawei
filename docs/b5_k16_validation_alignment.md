# B5：K16选模与新版main适配核对

## 已修改的运行协议

周期验证和最终validate/predict均默认EMA、Heun16、K16、seed42。每5轮验证，最后一轮也验证；FM训练方式、网络、条件、损失、LR=5e-5、100轮预算保持不变。

两组入口配置不改文件名，共同继承b5_base.yaml的sampling.samples=16：

| 实验 | 配置 | 新输出目录 |
|---|---|---|
| 华为从头训练 | b5_main_v2_scratch_meta.yaml | results/B5/main_v2_score60_k16_scratch_meta |
| 公开初始化后华为微调 | b5_main_v2_public_finetune_meta.yaml | results/B5/main_v2_score60_k16_public_finetune_meta |
| 通用华为微调 | b5_finetune_meta.yaml | results/B5/main_v2_score60_k16_finetune_meta |

生成仍逐份噪声累计取平均、复用条件编码。每窗Heun16/K16调用速度场512次，K1为32次；验证更慢，整段训练耗时取决于验证占比。固定seed、target_record_id、起始采样点和样本序号来生成噪声，批大小及顺序不改变噪声。

best.pt按K16下的正确总分选，best_r.pt按两任务r均值选，best_bonus.pt按Task2设备等权加分选，last.pt用于续训。它们只覆盖实际执行过验证的轮次，未评估轮次不会成为best。详细规则见[b5_score60_checkpoint_policy.md](b5_score60_checkpoint_policy.md)。

## 与main一致的部分

核对main=f607fb1（修正评分V3_10.11）。B5中的ecg12gen、common/preprocessing/data_spec/quality_rules/training_protocol配置、metadata配对/划分/QC/strict索引及总分脚本与该main无差异。B5训练运行器和模型保留在独立baselines/B5中。

训练读取main的strict去重target：1872窗/156条物理12导联记录。验证读取JointAnchorDataset的完整配对target：Task1 27配对/324窗，Task2机器40配对/480窗及体脂秤25配对/300窗。计数来自当前本地缓存，应在服务器prepare时再次确认。

target保留raw电压，模型输入输出只做冻结scale变换；不会为了新评分减去真实target均值。评分拼接12个10秒窗口成120秒完整记录，Task2两设备分别评分后1:1汇总。总分为50×(0.5r1+0.5r2)加两设备分别换算再等权的RMSE加分，最高60分。

网络仍为条件1D U-Net，使用同步I与人口学；main缓存中的异步context用于配对组织，未输入该基准。适配缓存与评分不代表已实现context融合。

## 服务器检查与运行

提交本机B5并推送后，服务器pull该分支。先检查，不启动训练：

```bash
cd /home/qht/huawei
python -m baselines.B5.prepare \
  --config configs/experiments/b5_main_v2_scratch_meta.yaml \
  --report results/B5/k16_scratch_preflight.json
python -m baselines.B5.prepare \
  --config configs/experiments/b5_main_v2_public_finetune_meta.yaml \
  --report results/B5/k16_public_preflight.json
```

两份报告应ready=true、sampling_matches_final_default=true，sampling为heun/16步/16份/seed42，selection_metric为competition_total_score，评分协议correlation50_bonus10_v1。确认人口学路径、sha、覆盖率以及两组scale/数据manifest一致。

两组直接采用[双实验说明](b5_main_v2_public_vs_scratch.md)的命令，新输出目录已经写在配置中。公开checkpoint可复用，不必重跑公开预训练。旧K1的last可以初始化新微调，但不能把其最佳记录直接resume到K16协议；通用微调配置允许旧华为权重初始化，配对E2配置仍只允许公开来源。

K16新运行可使用本组last.pt续训；保持原输出目录和已保存的best/best_r/best_bonus文件，避免只搬last而缺失历史候选。评分、采样、人口学、数据或损失协议变化时均须核对续训条件。

## 尚需实验检验的能力限制

1. 每窗10秒独立生成，训练中的辅助PCC按窗口计算，而正式评分在120秒拼接后计算。这两项目标不等价；当前没有跨窗口信息交换，窗口边界和窗口间慢走势仍需用实际预测检查。
2. 辅助重建使用FM插值状态的终点估计，实际测试通过从噪声积分生成。现有损失下降不保证积分后r同步改善。若要改变监督方式，应在稳定基准上另建实验、保持对照，不直接改当前损失。
3. 仅同步I和人口学可能不足以恢复每个胸导联独立的偏移、慢走势或幅值；这是输入信息与建模能力的限制，不能由配置修正证明解决。推理不能使用真实target的基线。
4. K16平均会改变波形与指标，需要同时看r、各设备胸导联RMSE和总分；不能仅凭r上涨认定幅值预测已改善。以前r上涨但RMSE偏大的情况不能只归因于缺少Transformer。
5. 本次两组各一个seed。先用一致K16协议完成公开预训练作用的对照；需要判断提升稳定性时再增加训练seed。比较历史K1选模实验时要标明规则改变，不能把全部差异归因于网络改善。

当前核对没有发现公共main接口或评分适配错误。以上属于后续实验方向，并非已经确认的性能根因。本次只调整K16验证协议、输出目录、预检信息、检查和说明，未训练或添加网络模块。
