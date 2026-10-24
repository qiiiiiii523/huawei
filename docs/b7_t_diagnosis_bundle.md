# 华为低r诊断包

脚本`scripts/pack_b7t_diagnosis.py`只读取原文件，在新目录生成诊断副本。CPU加载检查点，不调用CUDA，不训练、不修改正在运行的模型。

收集原T P/Q/A/B的配置、history、checkpoint_history、逐记录逐导联validation结果、调度与迁移记录；两组新T的阶段结果及日志；两个T工作树的实际Python代码、commit与未提交代码diff；患者划分、配对、QC及原始记录索引；三个缓存的完整manifest。训练/验证数据按组选择两个训练患者及低/中/高分验证记录，不复制测试波形。

导出原T四阶段best_r.pt的精简EMA诊断副本，保留模型结构、尺度、采样设置及来源信息，去掉模型副本、AdamW及RNG。这些副本用于推理诊断，不能续训；原始best/last均不改动。新T P的完整验证结果包含在包中，但不另复制两份新T权重。

波形诊断包括原始uV尺度、归一化标准差、幅值分位数、I与目标I差异、四肢体导联公式残差及QC。subset manifest中的NPZ路径为包内相对路径，可在本地直接读取；manifest_original保留服务器原路径用于溯源。没有保存预测波形的历史评估需要用这些EMA和样本重新生成；收集步骤本身不运行ODE生成。

## 服务器打包

用户先push本地分支的新提交，再在普通终端执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T_compact
git pull --ff-only gitee B7-T-compact

export B7T_DIAG="/home/qht/diagnostics/B7T_$(date +%Y%m%d_%H%M%S)"
mkdir -p /home/qht/diagnostics

if python -u scripts/pack_b7t_diagnosis.py --output "$B7T_DIAG"; then
  tar -C "$(dirname "$B7T_DIAG")" -czf "${B7T_DIAG}.tar.gz" "$(basename "$B7T_DIAG")" &&
  sha256sum "${B7T_DIAG}.tar.gz" &&
  ls -lh "${B7T_DIAG}.tar.gz"
else
  echo "收集失败：先检查报错，不上传未完成的诊断包。"
fi
```

预计几百MB量级，实际以ls为准。源run/缓存路径的默认值对应本次记录；路径不同可通过--run、--cache-run、--metadata等参数指定。输出目录必须是新目录，已有目录不覆盖。收集缺失的可选metadata会写bundle_report.json；必要权重或样本不存在时直接报错。

## bypy上传

在同一个终端（保留B7T_DIAG变量）：

```bash
bypy info
bypy mkdir B7T_diagnostics
bypy upload "${B7T_DIAG}.tar.gz" "B7T_diagnostics/$(basename "$B7T_DIAG").tar.gz"
bypy list B7T_diagnostics
```

首次授权按bypy提示完成；上传路径相对于bypy应用根目录。此命令只上传新生成的诊断归档，不删除服务器文件。需要提供给本地诊断时，将网盘里的tar.gz下载到本地并附到对话；仅上传网盘不会自动让本对话获得文件访问权。

不需要停止tmux或训练，也不需要复制全部原始华为数据。不要用精简EMA副本替换服务器best_r.pt/last.pt。
