# QC-v2：M2与g6_19m重新预训练

本次仅修改B7-U7-large与B7-T-compact中的数据准备、检查和启动保护，不修改模型/训练超参数。后续仅启动M2和g6_19m（19,398,702参数），不用双实验总控。原B7、原B7-T不更新；旧任务/权重保留为旧数据协议结果。

## 数据规则

- PTB-XL统一0=男、1=女，缺失值仍保留缺失掩码。拒绝未标记新版编码的旧PTB索引。
- 全12导联物理微伏float32波形做精确内容去重，优先保留test，其次validation，其次train；同级按record_id选择。只移除manifest中的副本，不移动患者到其他划分，也不删除原始文件。
- 全零/其他恒定导联、连续至少500个采样点完全相同的导联、单采样跳变>50mV的导联退出监督；I存在上述缺陷则排除整条记录。没有任何有效生成导联的记录也排除。
- 肢体公式RMSE>10uV的对应派生标签退出监督；I或II无效时其余四条肢体标签同步无效。
- >10mV幅度、>5mV跳变单独报告，除非达到上述拒绝规则，否则不自动删除。阈值属于本实验明确冻结的工程QC政策，不是医学诊断标准。
- FM、MAE、PCC使用已有lead_mask机制；模型不改。仅用去重、QC后的train拟合尺度。
- 新manifest保留数据格式v1，同时增加public_qc_version=B7-public-QC-v2；新P禁止旧公开缓存。缓存准备身份账本防止旧partial缓存混入v2。
- public_qc_report.json记录每条掩码、异常提示、被移除记录及重复组。masked目标的原始数据不被改写。
- PTBDB/华为不在这次公开10秒QC范围：可复用它们原始微伏NPZ，仅生成引用新尺度的manifest，不宣称完成了新的长窗QC。

修正数据后P从随机初始化训练，不使用旧P的--resume或--init-checkpoint。A直接初始化新P，Q初始化新P，B初始化新Q。旧权重再训练不能消除历史内容泄漏。

## 服务器：push后建立新工作树

在普通终端、已激活BioFlow环境执行。使用新工作树避免改变仍运行任务的代码；以下不停止任何已有进程。

```bash
git -C /home/qht/huawei_B7_U7_large fetch gitee
git -C /home/qht/huawei_B7_U7_large worktree add --detach /home/qht/huawei_B7_U7_large_qcv2 gitee/B7-U7-large
git -C /home/qht/huawei_B7_T_compact fetch gitee
git -C /home/qht/huawei_B7_T_compact worktree add --detach /home/qht/huawei_B7_T_compact_qcv2 gitee/B7-T-compact
```

若目标目录已存在，不要重复worktree add，也不要删除旧目录；核对git log和状态后使用已有的正确目录。

## 只准备一次共享数据，不启动GPU训练

普通终端执行；新公开NPZ约17GB，另有manifest/报告开销。长窗NPZ不复制，旧长窗缓存目录必须保留。

```bash
QC_DATA=/home/qht/huawei/data_cache/B7_public_qcv2
QC_PYTHON="$(command -v python)"
mkdir -p /home/qht/huawei/data_cache
tmux new-session -d -s b7-qcv2-data "cd /home/qht/huawei_B7_T_compact_qcv2 && '$QC_PYTHON' -u scripts/b7_public_qc_v2.py --data-root /home/qht/huawei/data --reuse-long-from /home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42 --output '$QC_DATA' > /home/qht/huawei/data_cache/B7_public_qcv2_prepare.log 2>&1"
tail -n 20 /home/qht/huawei/data_cache/B7_public_qcv2_prepare.log
```

完成时应出现[ready]。若准备被中断，用同样命令加--resume，只接受同一v2身份；旧v1的partial缓存不能续成v2。若长窗旧文件缺失，先解决缺失，不启动训练；也可使用prepare分别从原始PTBDB/华为重建并传入新版--scale-manifest。

准备后核查：

```bash
python - <<'PY'
import json
from pathlib import Path
base=Path('/home/qht/huawei/data_cache/B7_public_qcv2/data')
scales=None
for name in ('public10','ptb40','huawei'):
    m=json.loads((base/name/'manifest.json').read_text())
    assert m['public_qc_version']=='B7-public-QC-v2'
    if scales is None: scales=m['scales_uV']
    assert m['scales_uV']==scales
    print(name, len(m['records']), m['public_qc_version'])
q=json.loads((base/'public10/public_qc_report.json').read_text())
print('公开原始数/保留数:',q['input_records'],q['retained_records'])
print('重复组数:',len(q['duplicate_groups']))
print('无效输入或无有效目标排除数:',len(q['excluded_unusable']))
PY
```

## 只启动两组选定配置

确认数据准备成功及目标GPU有空间后再执行；下列GPU2和GPU0只是可编辑的分配，不保证当前服务器空闲。每个总控最多1个训练任务；两个总控不提供跨总控的共享GPU限流。

普通终端启动M2：

```bash
QC_PYTHON="$(command -v python)"
tmux new-session -d -s b7-m2-qcv2 "cd /home/qht/huawei_B7_U7_large_qcv2 && '$QC_PYTHON' -u -m baselines.B7.pipeline --data-root /home/qht/huawei/data --metadata /home/qht/huawei/metadata --ptb-root /home/qht/huawei/data/PTB-Diagnostic --reuse-data-from /home/qht/huawei/data_cache/B7_public_qcv2 --config-dir configs/u7_m2 --output results/B7-M2-qcv2 --gpu-p 2 --gpu-a 2 --gpu-b 2 --max-train 1 > m2_qcv2_launcher.log 2>&1"
```

普通终端启动约1940万参数T，只选g6_19m：

```bash
QC_PYTHON="$(command -v python)"
tmux new-session -d -s b7-t19-qcv2 "cd /home/qht/huawei_B7_T_compact_qcv2 && '$QC_PYTHON' -u -m baselines.B7.compact --preset g6_19m --data-root /home/qht/huawei/data --metadata /home/qht/huawei/metadata --ptb-root /home/qht/huawei/data/PTB-Diagnostic --reuse-data-from /home/qht/huawei/data_cache/B7_public_qcv2 --output results/B7-T-g6-qcv2 --gpu-p 0 --gpu-a 0 --gpu-b 0 --max-train 1 > t19_qcv2_launcher.log 2>&1"
```

不要使用run_b7_structures.py或compact_suite启动本轮：这些入口仍保留原有双实验功能，会额外启动未选定配置。

M2的P已改为batch64/30000步，validate_every=30000且validation_mode=final_only：跳过初始及中途全量验证，仅第30000步评估最终EMA；累计采样192万条次窗口，每500步保存恢复点。Q/A/B仍为batch32/各1000步。Transformer的g6_19m仍为P batch64/10000步，本次不改。以各分支已有配置为准，不统一更改它们已有的loss或采样批处理实现。新数据/尺度下结果应单独记录，不与旧协议的r直接混作同一组。

这是新版QC数据从随机初始化启动的M2新实验配置。不要将已启动的10000步实验直接改成30000步后--resume：恢复身份检查会拒绝变化。若30000步的新实验中断，保持该配置即可从其last.pt正常恢复。最终验证前的last.pt用于恢复，不代表已经按r选出中间最佳模型。

## 本地验证结果（2026-10-25）

- 实际重读全部66951条：去除87个重复副本，另排除5条不可用记录，保留66859条；保留内容指纹全部唯一。
- 另排除的5条：ptbxl:10565_hr、chapman:JS10616、JS42596、JS44076、JS45386。另有无效I的JS45519已计入重复移除，不重复计数。
- 保留PTB-XL train/validation/test=17417/2183/2198；Chapman=35874/4549/4638。
- 26条真实记录的小型缓存全流程通过，保留23条：验证重复跨split、无效I、恒定V5、局部平直V4、严重跳变V4，以及JS01052特殊头文件读取；尺度拟合和窗口加载成功。
- 两分支完整回归：B7-U7-large 63项、B7-T-compact 117项均通过。另包含无效导联不影响FM/MAE/PCC、缺失性别掩码、缓存身份拒绝、长窗NPZ不改写等验证。
- 尚未在服务器启动新版训练；本地未生成完整17GB公共缓存，仅生成了审计JSON与小样本缓存。
