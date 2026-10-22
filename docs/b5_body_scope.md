# body 输出范围四组对照

本实验不训练。冻结 FM/P、body 权重、β=0.75、Heun16/K16/seed42。
Task1 保持基础预测，机器保持原 FIR25；只有体脂秤路线变化。

|组|体脂秤投影|body 输出|
|---|---|---|
|R0_no_projection_no_body|否|无，辅助参照|
|R1_projection_no_body|是|无，辅助参照|
|C0_no_projection_II|否|仅 II，当前 B5-final|
|C0_five_no_projection|否|五肢体联合|
|C2_projection_II|是|仅 II|
|C3_projection_five|是|五肢体联合|

每条体脂秤记录仅调用 body 一次，读取原基础预测 II、当前可见 I、原始历史。
所有组复用 ΔII=body(原预测II)-原预测II，不向 body 输入投影结果。
投影使用现有固定真实可见 I、对五个预测肢体导联做最小二乘投影的公式。
五导联修正依次为 ΔII×[1,1,-0.5,-0.5,1]，I 和胸导联保持不变。
历史 B 去漂移只由当前 body 实现执行一次。缺失历史沿用通用 II 修正回退，状态写入报告。

## 远程运行

先准备与当前权重一致的纯 FM+P 基础预测缓存、验证数据和配置。
缓存不能是已经加过 context 的输出。服务器路径确认后替换下列示例路径。
同人群、同 QC 时添加 --verify-reference，要求 C0 复现文档成绩；改变人群时不要添加。

```bash
cd /root/autodl-tmp/huawei2.0
python -m baselines.B5_final.body_scope \
  --repository-root /root/autodl-tmp/huawei2.0 \
  --checkpoint /root/autodl-tmp/huawei2.0/models/base.pt \
  --body-checkpoint /root/autodl-tmp/huawei2.0/models/body.pt \
  --config /实际路径/validation.json \
  --baseline-evaluation /实际路径/纯基础预测缓存 \
  --output /root/autodl-tmp/huawei2.0/results/body_scope_run001 \
  --device cuda
```

如无缓存，先按现有入口生成（config 必须是已调整好实际数据路径的验证配置）：

```bash
python -m baselines.B5_best.validate \
  --checkpoint models/base.pt --config /实际路径/validation.json \
  --output-dir results/plain_base_run001 --device cuda
```

输出目录必须不存在，失败报告保留，重试换新目录。
实验保存六套预测数组，磁盘需求约为单份基础预测数组的六倍，另加评分报告。
权重 SHA256、训练来源和缓存采样设置沿用 B5-final 的严格检查。

## 查看结果

- summary.csv：Task1、机器、体脂秤、Task2、总平均 r，以及相对 C0 的增量。
- contrasts.csv：无 body 时投影收益、仅 II/五导联条件下投影收益、两种投影条件下的扩展收益。
- 各组 task1/task2 评分报告：沿用当前完整120秒、逐目标QC和目标记录等权口径，包含逐导联结果。
- 各组 pair_metrics.csv：逐患者/配对逐导联 r，用于排查改善是否集中在少数患者。
- postprocess_records.csv：历史使用与回退状态；protocol.json/final.json：权重、实际路径、数据指纹及复现检查。

先看 C0 是否可信复现，再看 C0_five 与 C0、C3 与 C2；同时检查逐导联得失。
不自动根据验证分数替换部署模型，也不把小幅领先视作稳定收益。
本实验尚未检查训练集真实肢体关系误差；有收益后，联合训练前需补该诊断。

本地数值测试：`python -m unittest discover -s tests -p test_body_scope.py -v`。
