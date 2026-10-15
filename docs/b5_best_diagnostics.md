# B5-best诊断运行之后：接下来怎么做

适用进度：已经在服务器运行过`run_b5_best_diagnostics.sh`。本文从运行结束后开始，不再重复创建worktree、更新代码或启动诊断。

原数据与历史结果根目录：`/home/qht/huawei`。新诊断报告位于该目录的`results/B5-best/diagnostics_时间戳_进程号`，不是B5-best训练记录。

## 第一步：定位刚才的报告，并确认运行完成

在服务器终端执行：

```bash
cd /home/qht/huawei

run=$(ls -dt /home/qht/huawei/results/B5-best/diagnostics_*/ | head -n 1)
run=${run%/}

printf '本次报告目录：%s\n' "$run"

cat "$run/diagnostics.json"
```

这里自动定位最近修改的诊断目录；`run=${run%/}`去掉末尾斜杠。后面命令沿用这个终端的`run`变量。若换了新终端，先重新执行这一步。

成功应看到：

```json
{
  "completed": true,
  "training_started": false,
  "inference_started": false,
  "deployment_model_changed": false,
  "beta1_mean_r": 0.69262935
}
```

上面仅展示关键字段，数值允许很小浮点差异。`completed:true`表示整套诊断结束；只有`preflight.json`并不代表诊断完成。

若提示找不到`diagnostics.json`，先查看同名日志：

```bash
cat "${run}.log"
```

此时先处理未完成原因，不打包为成功结果，也不重复启动训练。若自动定位到了后一次失败目录，可用下面命令列出历史目录，然后手动指定已完成的那一个：

```bash
ls -dt /home/qht/huawei/results/B5-best/diagnostics_*/

# 把下一行替换为上面列表中真实存在、已经完成的目录。
run=/home/qht/huawei/results/B5-best/diagnostics_实际时间戳_实际进程号
```

## 第二步：查看主要结果（可选，也可以直接进入第三步打包）

```bash
cat "$run/report.md"

cat "$run/beta_scores.csv"

cat "$run/integration_scores.csv"
```

这三份文件分别给出：

| 文件 | 回答什么 |
|---|---|
| `report.md` | 诊断完成情况、基准r与阅读说明 |
| `beta_scores.csv` | 慢系数0.75、1、1.25对最终r的影响；1是当前稳定方案 |
| `integration_scores.csv` | 已有B5-S50三组best_r的同QC结果；缺失数组会明确标记，不重新生成 |

需要查看图时，把整个`plots`文件夹下载到本机。图按低/中/高诊断r选代表记录，包含V1/V3/V5/V6的慢走势、原始局部、快残差局部、I锚点心搏叠加与慢曲线频谱。

其余CSV先保留，不需要你逐个手动解读：

- `component_record_lead.csv / component_lead_summary.csv`：快慢强度、慢曲线振荡及逐导联差异。
- `morphology_record_lead.csv / morphology_lead_summary.csv`：峰谷幅度、时间偏差、半高宽与局部形态。
- `qrs_anchor_audit.csv`：I锚点数量，用于发现自动心搏检测明显不可靠的记录。
- `boundary_record_lead.csv / boundary_lead_summary.csv`：接缝附近与窗口中部误差。
- `preflight.json / diagnostics.json / plot_manifest.json`：来源路径、完成状态和图片对应记录。

**当前不要因为某个beta在验证集最高，就直接改正式配置或替换稳定模型。** 这三个系数是方向探针，需结合逐记录和形态结果判断。

## 第三步：打包报告，发回来分析

仍在同一个终端，执行：

```bash
archive=/home/qht/huawei/B5_best_diagnostics_$(date +%Y%m%d_%H%M%S).tar.gz

tar --exclude='*.npy' --exclude='*.pt' \
  -czf "$archive" \
  -C "$(dirname "$run")" \
  "$(basename "$run")"

ls -lh "$archive"
```

最后一行会打印真实压缩包路径和大小。下载这个tar.gz，发送即可。它包含诊断CSV、JSON、说明和图片；不需要权重、预测数组、原始ECG，也不需要重新打包全部B5-S50历史训练。

想额外确认压缩包能读，可以执行：

```bash
tar -tzf "$archive" > /dev/null
```

没有报错即目录可读。如果`run`变量为空或换过终端，先回第一步，不能直接复制第三步单独运行。

## 第四步：根据诊断决定下一项实验，不提前启动新训练

收到上述压缩包后，按以下逻辑选下一步：

| 诊断证据 | 对应下一项实验 |
|---|---|
| 相对慢起伏过强/过弱，beta探针在多数记录方向一致 | 小范围组合系数实验，再做独立复核 |
| 预测慢曲线夹带真值没有的快速振荡 | 针对最终组合的慢CNN监督/约束对照 |
| V1/V3峰宽、形状或时序问题明确 | 选择一处高分辨率I编码或FM解码模块改动 |
| V5/V6快对应好、长慢走势仍差 | 优先慢预测器目标或时间建模改动 |
| 接缝误差确实集中且影响明显 | 重叠预测或相邻窗口对照 |

这些新实验的代码与运行参数需在诊断确认后明确实现。**当前没有一条可直接照抄的新训练命令，不应把尚未实现的功能当作已有配置运行。**

当前稳定方案仍为`原B5快残差＋S50-F预测慢曲线`，只按两个任务平均r评价；不重新做M，不把RMSE加回评分，不因诊断而改动旧训练目录。
