# M2-L：七通道FM，缺失11导联PCC监督

本轮只实现监督范围修改，没有实现循环重建、第二个模型或Transformer模块。
原M2/M3/小U7配置保持七导联PCC。选择`configs/u7_m2_l`启用本轮新实验。

## 损失与数据

`L=FM7(II权重5)+MAE7(II权重5)+0.1*PCC11(导联等权)`。
七条生成导联仍为II、V1–V6；干净原始I直接输入，不加噪、不生成I。
物理单位uV下用I和预测II构造四条派生导联，PCC比较真实标签的缺失11条。
公式计算可微，III/aVF等相关性误差能反传给预测II，不增加输出参数/噪声通道。
输出关系仍严格成立至浮点精度。

PCC使用训练终点估计，仍不是K16自由生成评分。P用完整真实10秒；长窗用中央15–25秒，
FM/MAE用40秒真实有效区，padding排除。PCC按有效样本/导联等权，不声称与最终
pair/target/group分层宏平均完全相同。I不计入PCC，也不通过复制I虚增成绩。

数据读取器仅在`loss.pcc_leads=11`时从原NPZ额外返回真实12导联、12导联QC、冻结尺度。
不重新prepare，不覆盖缓存。先反归一化进行导联组合，再按每条固定训练尺度归一化以保持
数值稳定与既有目标方差阈值单位；PCC本身不受正比例缩放影响。
真实标签不被I/II合成标签替代。QC不一致、缺少真实12标签、尺度无效时明确拒绝；
无有效PCC导联/常数目标贡献0，非法目标按QC屏蔽，数值梯度检查已覆盖。

## 新实验而非旧任务断点迁移

推荐从已完成的M2公开P EMA初始化：架构、尺度不变，只加载EMA，重置优化器和新实验局部步数。
checkpoint记录源权重SHA、源stage/步数/loss、optimizer_reset。不要修改原checkpoint身份。
新任务自己的`--resume`仍完整恢复模型、EMA、优化器、随机状态、步数。
P局部0对应从旧M2 P的10000步EMA出发，不是随机初始化，也不是原任务总预算仍10000步。

| 阶段 | 初始化 | batch | 新增步数 | 学习率 | 全量验证 |
|---|---|---:|---:|---:|---|
| P_L | 已完成M2 P EMA | 64 | 5000 | 3e-5 | 仅末尾 |
| A_L | P_L末步EMA | 32 | 1000 | 1e-5 | 仅末尾 |
| Q_L | P_L末步EMA | 32 | 1000 | 1e-5 | 仅末尾 |
| B_L | Q_L末步EMA | 32 | 1000 | 1e-5 | 仅末尾 |

λr先保留0.1、II权重保留5。本轮不是增加II系数或频带MAE；这些可另行调参。
K16、Heun16、记录batch4、ODE chunk16、每500步保存均保留。
每阶段仅末尾验证，`best_r.pt`在本配置中指该阶段末尾EMA；没有沿用旧P最佳分数来阻止保存新阶段权重。
比较旧0.74019与新P_L末尾结果，不能预先承诺提高。此后华为A/B决定实际迁移效果。

## 服务器运行

旧B7、M2/M3正在运行时不要在其目录git pull更新模型包，否则严格恢复身份会改变。
推送本地B7-U7-large最新提交到gitee后，在服务器独立checkout中使用新代码：

```bash
cd /home/qht/huawei_B7_U7_large
git fetch gitee B7-U7-large
git worktree add --detach /home/qht/huawei_B7_M2_L gitee/B7-U7-large
tmux new -s b7_m2_l
```

若新目录已存在则进入已有目录，不重复worktree add。不修改旧训练checkout。
新tmux中运行如下命令，GPU参数按当前资源选择；示例使用GPU0且本套件串行，
未施加40GiB停止阈值，也不做启动GPU短测。其他任务的显存仍需计入。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_M2_L
mkdir -p results/B7-M2-L
set -o pipefail
python -u -m baselines.B7.pipeline \
  --data-root /home/qht/huawei/data \
  --metadata /home/qht/huawei/metadata \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --reuse-data-from /home/qht/huawei_B7/results/B7/auto_seed42 \
  --config-dir configs/u7_m2_l \
  --init-p-checkpoint /home/qht/huawei_B7_U7_large/results/B7-M2-M3_gpu2/M2_seed42/P/best_r.pt \
  --output results/B7-M2-L \
  --gpu-p 0 --gpu-a 0 --gpu-b 0 --max-train 1 \
  2>&1 | tee -a results/B7-M2-L/launcher.log
```

同一命令可恢复新实验，自动对已有阶段添加--resume；输出目录必须独立于旧M2。
外部初始化checkpoint路径与SHA纳入总控指纹，恢复期间不要替换该源文件。
GPU分配可以修改，训练配置/源码/数据身份不能随意修改。
如要A/Q并行可设置--max-train 2，显存不足仍会OOM，不自动改变batch。

`--dry-run`只打印P→(A,Q→B)命令，不prepare、不训练。
新训练启动日志应显示`pcc_leads=11`、`parameters=11477663`。
查看状态/日志：

```bash
cat results/B7-M2-L/pipeline_status.json
tail -n 20 -f results/B7-M2-L/logs/P.log
```

本机仅CPU回归测试，未在实际数据/GPU训练或确认效果。

## 循环重建与分阶段模型是两个独立方向

当前A版：一个FM联合生成II和六条胸导联，再用公式补齐；本轮实现的是A版的PCC11。

可选B版：一个FM仅以I为条件生成II；公式恢复六条肢体导联；第二个FM以预测肢体导联为条件生成胸导联。
合计仍为1+6条生成，但两个FM的训练和推理分阶段执行。B版不必附加循环重建。
训练第二阶段不能只使用完美真实肢体导联后直接期待适应推理误差；应采用训练患者上的交叉拟合预测
或其他有明确记录的预测条件训练方式，不能让生成训练条件的模型偷看验证/测试患者。

论文中的循环：I→生成II→一个学习的逆变换→重建I，比较重建I与原始I，梯度穿过逆变换反馈给正向模型。
这不是把原始I复制回去。复制会使loss恒零；把生成II detach也不能给正向生成器反馈。
用于FM需要可微分采样；现有integrate/sample标记no_grad，不能直接用于循环loss反传。
可以额外训练II→I反向FM，然后用少量可微ODE步/单次噪声构造循环损失，保持正向真实II监督。
冻结逆模型参数时仍需保留它对输入II的梯度，不可包在no_grad内。
两种变换并不天然唯一可逆，两个模型可以配合还原I却仍生成不准确II，因此循环不是真实性保证。
增加循环需要单独验证收益和计算开销；本轮未添加这个第三条训练通路。
