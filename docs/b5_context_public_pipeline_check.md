# ContextPublic完整入口本地检查（2026-10-23）

本轮已补齐数据准备、两阶段训练、成对基准复评、完整组合评估、比较汇总、机器新记录推理和Bash启动入口。
此前审计阶段的`training_ready=false`不再代表缺少代码；现在由服务器实际prepare产生的study/preflight控制是否可启动。

本地隔离CPU依赖下完成：

- 11项测试全部通过，无跳过。
- 实际V6加载B5后，零初始化前向输出一致；启用新分支后全无效历史正确回退。
- 实际反向检查原主干及历史编码器存在梯度。
- 成对噪声精确相反，记录键生成不受batch顺序影响。
- 小型合成两阶段训练中，公开epoch中断后恢复，最终模型、EMA和更新数与不中断训练完全一致。
- 一条真实PTB记录只读标定/QC执行成功，输出12导联500Hz数组；未启动真实ECG模型训练。
- prepare/references/train/evaluate/compare/predict的CLI帮助均正常导入；Bash脚本语法通过。

仍须服务器执行：全量真实数据prepare与QC、固定G120人群对齐检查、4份成对K16完整参考复评，再开始真实训练。
本地检查不构成GPU运行、全量患者缓存或真实比赛分数已验证的声明。

运行说明以`docs/b5_context_public.md`为准。profile保留显式采样协议，prepared study绑定源资产和代码SHA；中途换源码不能无缝续跑旧study。
