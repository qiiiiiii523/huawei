# 已迁移B7-T：P只做最终验证，P在GPU0、长窗在GPU1

新增独立入口`scripts/b7t_p_final_only.py`，适用于当前`auto_bs32_sparse_seed42`：P尚未完成，A/Q/B尚未启动，核心代码与原migration.json一致。它不修改baselines/B7模块或项目共用的configs，也不修改旧run。P的专用配置把validate_every设为总更新次数10000，每500步恢复点保持不变；Q/A/B配置完全复制原值。

若停止时P有未完成验证，必须显式传入--skip-pending-validation。此时保留训练步数、AdamW/EMA/RNG及真正的last_validated_update，并另外记录skipped_validation；不产生虚假的验证分数。若6000步已评估保存，工具会保留结果。10000步最终验证不能跳过。

迁移后的运行必须通过该脚本的run子命令启动/恢复：它在当前进程内为原pipeline选择新run自己的配置文件，并将脚本及配置纳入fingerprint。原pipeline的train、完整身份核验、完成检查、P→(A,Q→B)依赖与错误隔离继续使用。不要对新目录直接调用普通pipeline入口，它会选项目共用配置而身份不匹配。

## 先同步脚本和CPU测试

本次只新增脚本、测试和此说明，没有在侧对话中提交Git。请先在本地按原方式提交并push这三个文件到Gitee；不会改变正在运行的模型模块或共用配置。

服务器普通终端：

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T
git pull --ff-only gitee B7-T
CUDA_VISIBLE_DEVICES="" python -m unittest scripts.test_b7t_p_final_only -v
```

必须看到OK才继续。本地只完成语法检查；本地PyTorch依赖读取被当前权限限制，未运行这些真实状态测试，也没有执行服务器迁移。

## 停止B7-T总控

在普通终端执行`tmux attach -t b7-t`，选择真正运行B7-T总控的窗口（不是tail查看窗口），按一次Ctrl+C并等待返回命令提示符。不要停止b7会话。脚本还会检查旧进程及源run锁，仍在运行会拒绝迁移。

## tmux b7-t内迁移并启动

```bash
conda activate BioFlow
cd /home/qht/huawei_B7_T
B7T_OLD=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_sparse_seed42
B7T_NEW=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_final_only_seed42

python -u scripts/b7t_p_final_only.py migrate \
  --source "$B7T_OLD" --output "$B7T_NEW" \
  --skip-pending-validation
```

必须看到status=MIGRATED才启动。新目录必须不存在；发现已启动A/Q/B、best/last不一致、配置/数据变化或代码不符时工具拒绝，不要删旧文件或强行跳过检查。

```bash
set -o pipefail
python -u scripts/b7t_p_final_only.py run \
  --output "$B7T_NEW" \
  --data-root /home/qht/huawei/data \
  --metadata /home/qht/huawei/metadata \
  --ptb-root /home/qht/huawei/data/PTB-Diagnostic \
  --gpu-p 0 --gpu-a 1 --gpu-b 1 --max-train 2 \
  2>&1 | tee -a "$B7T_NEW/pipeline.log"
```

默认A/Q可以同时占GPU1；B在Q完成后接续。若GPU1资源只够单个训练，将max-train改为1。当前P仍在GPU0。后续中断只重跑run命令，不能重复migrate。离开tmux用Ctrl+B后按D。

旧raw cache位于原`auto_bs32_seed42/data`，工具从旧migration.json取得并校验，不把当前sparse目录下的空data当作缓存。旧run及原缓存保留，不重新prepare。

普通终端看进度：

```bash
B7T_RUN=/home/qht/huawei_B7_T/results/B7-T/auto_bs32_final_only_seed42
python -m json.tool "$B7T_RUN/pipeline_status.json"
tail -n 30 "$B7T_RUN/logs/P.log"
```

预计启动日志显示validate_every=10000、checkpoint_every=500，并从停止时最新已保存训练步恢复。若这次跳过6000步验证，最近完成验证仍显示5500步，这是保留真实评估历史，不是错误。
