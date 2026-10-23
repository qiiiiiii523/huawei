# Batch32与批量验证

用户确认P/A/Q/B全部训练batch32，同时优化验证。三份阶段配置统一training.batch_size=32；保持原learning_rate、更新次数、验证间隔、FP32、loss和评分口径。样本预算因此增加：P为32×10000=32万条次，A/Q/B各32×1000=3.2万条次，不是原训练预算的等量对照。长窗batch32保存激活估计约14.6GiB/训练进程，另有反向临时张量、cuDNN、CUDA及优化器开销，需服务器实测，不能保证双进程总峰值。

最终复核还将PTB患者目录过滤及包含患者目录的唯一ID规则补入默认prepare，防止从零准备时再次遇到root副本重名；复用旧成功manifest的路线不改数据身份。最新验证为45项核心测试及5项独立恢复测试全部通过。

验证最多8条等长记录一起进入predict_records，长记录按原中央10秒核心区拼接；不同长度先flush，不把不同患者信号接到一起。K16保持16份原keyed噪声；将(record,k)轨迹分批，每批至多32条，保持原sample-index和累加顺序。实际网络批内算子可能引入浮点舍入差异，不承诺逐bit一致。原QC、逐导联/目标/配对聚合和best_r规则不变。

日志新增train启动配置、每50次更新进度，validation开始与每批scored/scanned进度。旧checkpoint未含批处理字段时默认按1加载。

## 服务器切换

先在b7会话原总控窗口及PTB恢复窗口分别Ctrl+C停止（不要操作B5窗口）。在普通终端检查：

```bash
pgrep -af 'baselines\.B7\.(pipeline|train|prepare)|scripts/recover_b7_ptb.py'
```

确认旧B7任务均已停止，保留旧auto_seed42、initial/last/best和全部缓存。不要用killall/pkill python。将最新提交推送Gitee后再在服务器更新代码；在运行中的P/A上切换源码会破坏身份校验。

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
git fetch gitee
git switch --detach gitee/B7
```

在b7 tmux新窗口执行整段：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
mkdir -p results/B7/batch32_seed42
set -o pipefail
(
  CUDA_VISIBLE_DEVICES=0 python -u scripts/check_b7_batch32_gpu.py &&
  python -u -m baselines.B7.pipeline \
    --data-root /home/qht/huawei/data \
    --metadata /home/qht/huawei/metadata \
    --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
    --reuse-data-from results/B7/auto_seed42 \
    --output results/B7/batch32_seed42 \
    --gpu-p 0 --gpu-a 0 --gpu-b 0
) 2>&1 | tee -a results/B7/batch32_seed42/pipeline.log
```

GPU检查先对10秒/40秒训练batch32执行真实forward/backward/AdamW及K16批量采样，输出峰值显存；检查失败不启动总控。没有服务器GPU实测结果前不声称一定装得下；同卡两个长窗训练若显存不足，重新运行总控加--max-train 1按最近checkpoint恢复，不改batch。

--reuse-data-from只读取原三份成功manifest，优先使用ptb_recovery/data的修复PTB缓存；验证scale相同、数据文件存在，不prepare。若旧缓存尚未完成，入口会报错，保留现场。新输出必须没有旧权重；本次P从零重训（不自动继承旧小batch的P），A/Q加载新P，B加载新Q。若只想延续旧P属于不同实验，需显式init并重置优化器，不能冒充原resume。

原恢复脚本不再启动。离开tmux用Ctrl+b后d。新进度在batch32_seed42/pipeline.log与logs/P.log；全部阶段最终统一在pipeline_status.json汇总，无需单独route日志。以后恢复同一batch32实验，代码、配置和数据不变，直接重跑pipeline命令复用/恢复；GPU检查可省略。
