# 实验记录模板

每次 B0/B1/B2 实验复制本模板到受控实验记录系统或团队日志中；不要把患者级预测、原始 ECG 或大体积数组提交到 Git。

## 基本信息

- 实验编号：
- 日期与执行人：
- Git commit：
- 分支与标签：
- 模型名称/版本：
- 任务：task1 / task2
- 目的与唯一变量：

## 固定协议确认

- `metadata/subject_split.csv`：文件版本或哈希
- 数据划分：fixed subject-level 80% train / 20% validation
- 随机种子：42
- ECG：μV、500 Hz、10 秒窗口、10 秒步长
- 导联顺序：I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6
- 训练数据门控：`pair_status=paired`、usable；不包含 review/unmatched
- 预处理协议：`configs/preprocessing.yaml` v1（train-only scale；train、validation、推理使用同一冻结实例）
- 是否使用 PPG / acceleration：

## 训练参数

- batch size：
- epoch：
- optimizer / learning rate / weight decay：
- 损失函数：
- checkpoint 选择规则：仅按 validation 指标

## 结果

- task1 r1 或 task2 r2：
- 12 导联平均 Pearson r：
- 12 导联平均 RMSE（μV）：
- 任务二缺失导联平均 RMSE（μV）：
- V0 报告路径：`overall_metrics.csv`、`lead_metrics.csv`、`report.md`；任务二另附 `task2_subject_metrics.csv`、`task2_device_metrics.csv`。
- 异常、失败或可比性备注：
