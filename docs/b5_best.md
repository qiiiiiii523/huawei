# B5-best：control100 FM＋SlowPCC20稳定组合

> 当前默认验证已启用可靠输入I过滤。本文0.698216等参考分数属于旧目标QC范围，保留为历史记录；权重没有改变。新分数需重算，不能与这些旧分数直接比较。更新与CPU重算见[同步操作步骤](b5_sync_reliable_i.md)。

## 当前默认方案

2026-10-17，将已完成交叉评估和真实完整推理复现的组合提升为B5-best默认：

```text
120秒同步I＋人口学
  ├─ control100 FM（原条件1D U-Net），Heun16/K16/seed42 → p
  └─ SlowPCC第20轮慢CNN，50秒I上下文、预测中心10秒 → 拼成s

缺失11导联 = p − L(p) + 0.75 × s
L：完整120秒上的501点居中移动平均
```

FM来自B5-scratch的control第100轮EMA，不是慢条件注入模型。慢CNN仍是S50-F网络，使用原慢标签/F监督并新增最终组合PCC，FM在该慢CNN训练期间冻结。此次更新不改变网络结构、条件、滤波、插值、采样或评分，不重新训练。

测试只需要完整120秒、500Hz、原始μV的同步I与可用人口学。不使用真实待补导联、目标均值、目标慢曲线或设备历史context。S50的50秒上下文来自同步I，记录边缘复制；这不是实时因果模型。I输出保持FM的I头，正式r只计缺失11导联。

## 已验证成绩与来源

|评估|Task1 r|Task2 r|平均r|
|---|---:|---:|---:|
|seed42交叉组合|0.7023541975|0.6940771018|0.6982156497|
|seed43噪声复核|0.7024726933|0.6934860629|0.6979793781|
|完整部署入口复现，seed42|0.7023541744|0.6940770972|0.6982156358|

部署入口已返回`reference_matched=true`。同批验证患者的两次噪声复核不等同于独立训练或独立患者复现。

默认配置为`configs/b5_best.json`。它固定以下完整包及SHA，而不是把两种不同训练checkpoint格式交给旧导出器：

```text
/home/qht/huawei/results/B5-best-cross/seed42_20261017_094115_417122/control100__pcc20.pt
SHA256=a8d32cd598f73a0d3335ff4453da9168eccbc518a9b508ae81fdb8babcce254b
```

部署复现报告：

```text
/home/qht/huawei/results/B5-best-cross/deployment_check_20261017_100949_439149
```

包已包含FM EMA、慢CNN、尺度、人口学schema、β和采样参数。默认推理不需要原训练checkpoint、prepared或训练缓存；请保留这一个模型包。原两份训练权重路径在配置components中仅作来源记录。

## 1. 推送与服务器更新

本机：

```powershell
Set-Location -LiteralPath 'C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best'
git push origin B5-best
```

确保提交已同步到服务器使用的gitee。在已有B5-best工作树更新：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best
git status --short
git fetch gitee B5-best
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

如果工作树尚不存在：

```bash
cd /home/qht/huawei
git fetch gitee B5-best
git worktree add --detach /home/qht/huawei_B5_best FETCH_HEAD
```

不在正在训练的其他工作树切分支。更新冲突时保留本地文件，不执行reset --hard。

## 2. 立即使用默认模型验证

新默认包已经真实复现过，无需因为Git配置升级再训练。需要在此稳定工作树复核时执行：

```bash
(
set -euo pipefail
cd /home/qht/huawei_B5_best
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
run=/home/qht/huawei/results/B5-best/verify_default_$(date +%Y%m%d_%H%M%S)_$$

python -u -m baselines.B5_best.validate \
  --config "$config" \
  --device cuda \
  --verify-reference \
  --legacy-target-qc-only \
  --output-dir "$run" \
  2>&1 | tee "${run}.log"

cat "$run/correlation_score.csv"
echo "结果目录：$run"
)
```

上面的命令显式复现历史目标QC范围。省略checkpoint时仍核对默认包SHA、组件来源、β、采样及原参考r。新验证请去掉--legacy-target-qc-only和--verify-reference，或使用新协议重算得到的同权重--reference-report配合--verify-reference。不能拿旧参考r要求新范围复现。120秒、目标记录/导联规则、Task2设备等权及RMSE仅诊断保持不变。

数据配置从原服务器训练run_config读取；其旧选模字段在data_only模式下不参与当前评分。无需把数据复制到这个工作树。

## 3. 新记录默认推理

将占位输入换成真实文件：

```bash
cd /home/qht/huawei_B5_best
CUDA_VISIBLE_DEVICES=1 python -m baselines.B5_best.predict \
  --anchor-npy /path/to/new_I_uV.npy \
  --metadata-json /path/to/new_metadata.json \
  --record-key new_record_001 \
  --output /home/qht/huawei/results/B5-best/new_record_001 \
  --device cuda
```

无需传checkpoint、β、K或步数。I为[60000]或[1,60000]，500Hz、μV。人口学age、sex/gender、height、weight允许缺失，也可省略metadata-json。输出为[12,60000]原始μV，inference.json记录模型SHA和采样设置。稳定record-key用于固定推理噪声。

## 4. 保存独立稳定副本（可选）

如果要把模型从实验目录复制到稳定目录，使用已有脚本：

```bash
cd /home/qht/huawei_B5_best
sed -i 's/\r$//' scripts/run_b5_best.sh
bash -n scripts/run_b5_best.sh
bash scripts/run_b5_best.sh --gpu 1
```

默认创建`/home/qht/huawei/results/B5-best/stable_control100_pcc20_v3/`，包含model.pt、model.json及evaluation。复制按字节保留原包，SHA不变；随后从副本完整验证，不训练，不联网。已有目录不会覆盖，可用`--root`指定新目录。

单独只复制模型、不评估：

```bash
python -m baselines.B5_best.bundle \
  --output /home/qht/huawei/results/B5-best/stable_control100_pcc20_v3/model.pt
```

以后若移动模型或删除实验目录，可显式传该副本：

```bash
python -m baselines.B5_best.predict \
  --checkpoint /home/qht/huawei/results/B5-best/stable_control100_pcc20_v3/model.pt \
  --anchor-npy /path/to/new_I_uV.npy \
  --record-key new_record_001 \
  --output /home/qht/huawei/results/B5-best/new_record_001 \
  --device cuda
```

默认配置仍指向原已验证的完整包路径，不会自行扫描、挑选新权重或默默回退旧模型。

## 5. 旧模型及历史诊断

- 显式`--checkpoint`继续支持旧v1/v2完整包；旧v1仍按β=1，更新代码不会改写已有包。
- 旧原FM＋原S50-F的β=0.75配置独立保留在`configs/b5_best_legacy_beta075.json`。
- 既有diagnose及run_b5_best_diagnostics.sh分析的是旧原FM/S50-F历史数组，已绑定旧配置，不拿新参考r核验旧结果。
- check_beta_seed默认分析当前新组合；复现旧组合时传`--profile configs/b5_best_legacy_beta075.json`。
- 原始B5-U＋旧S50候选导出须显式传旧profile与`--candidate`；scratch或SlowPCC训练checkpoint不是普通B5-U/S50训练格式，不能仅靠换路径交给旧导出器。

本分支后续B5-U训练仍沿用原网络和FM损失，按main_correlation保存best_r.pt/last.pt；这次只更新默认部署来源，未改变训练程序。

## 本地验证边界

32项CPU功能与相关回归检查通过，Bash语法检查通过。覆盖默认包SHA与来源绑定、缺包拒绝、按字节复制、源组件删除后的推理、默认predict/validate路由、旧v1/v2兼容及历史诊断。

本机没有服务器真实模型包与GPU环境，只做CPU合成功能及回归检查，不训练。实际完整生成复现依据为用户提供的服务器结果，数据和分数如上；不把本地合成测试当作ECG性能结果。
