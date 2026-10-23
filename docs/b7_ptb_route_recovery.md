# P/A运行中恢复被阻止的PTB/Q/B路线

服务器PTB-Diagnostic扫描550个头，发现同名`patient279/s0533_re.hea`和根目录`s0533_re.hea`。自动索引把根目录文件也当患者记录，导致ID重复。此处不更改运行中的B7包；新增独立`scripts/recover_b7_ptb.py`。

索引规则明确为PTB根目录下`patient数字/记录.hea`，root级头列入排除审计，不删除原文件、不宣称其波形已证明相同；遇到其他嵌套结构停止而不猜测。记录ID包含患者目录，继续沿用既定按患者hash划分。审计保存在`ptb_recovery/index_audit.json`。

## 服务器操作

先把新脚本提交推送Gitee。在普通服务器终端只取这一个文件，不切换运行中代码版本：

```bash
cd /home/qht/huawei_B7
git fetch gitee
git checkout gitee/B7 -- scripts/recover_b7_ptb.py
```

进入原b7会话并Ctrl+b、c新建窗口。在新窗口执行：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7
mkdir -p results/B7/auto_seed42/ptb_recovery
set -o pipefail
python -u scripts/recover_b7_ptb.py \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --output results/B7/auto_seed42 \
  --gpu 0 \
  2>&1 | tee -a results/B7/auto_seed42/ptb_recovery/route.log
```

新脚本在原总控Q/B已blocked时才能运行，加独立进程锁防止重复恢复。准备新缓存`ptb_recovery/data`，复用公开P的scale；成功后等待P completed，核对原代码、配置、数据和有效权重，再启动Q。Q成功后华为缓存就绪即可启动B，不等待A。GPU0，两路训练最多为A与Q/B。P失败则停止；准备失败或训练失败保留日志，不跳过错误；再次启动脚本可复核半成品缓存并恢复已有last.pt。

原总控的pipeline_status.json会继续显示Q/B blocked，因为原任务不自动重试；这不是恢复脚本的状态。Q/B以独立`ptb_recovery/status.json`、`route.log`及Q.log/B.log为准。最终权重仍放原`auto_seed42/Q`、`auto_seed42/B`。

普通终端检查：

```bash
tail -n 20 -f /home/qht/huawei_B7/results/B7/auto_seed42/ptb_recovery/route.log
cat /home/qht/huawei_B7/results/B7/auto_seed42/ptb_recovery/status.json
```

不要停止原总控；不要在同一Q/B目录同时手动启动其他训练。脚本默认等待上游最多30天，可用--wait-seconds设置。华为准备失败则不会启动B。未映射华为人口学的既定限制不变。

## 已验证范围

本地5项独立测试通过：root头排除审计与原文件保留、未知嵌套拒绝、P失败停止等待、拒绝与原Q并跑、GPU0准备→等待P→Q→B调用顺序。调度测试使用模拟进程，不是真实GPU训练。

本地真实PTB根目录548条正式记录均保留，uniqueID548，train454/validation43/test51；未读用户服务器550条全部波形，不预测其过滤后数量。相对0113fbc，baselines/B7与configs均未改动，因此此恢复脚本本身不改变B7训练code_sha。只取脚本的部署方式也避免更新任何模型源码。
