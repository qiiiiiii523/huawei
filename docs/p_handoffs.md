# M2与19.4M B7-T公开P交接

两个模型分别打包，不能混用同名baselines代码。M2精确参数11,477,663，
B7-T g6_19m精确参数19,398,702；打包器严格核验参数量、P阶段完成记录、最后一步与选定权重。
当前已知公开P验证r分别0.7401874425、0.7219846581。没有发华为A/A1/B权重。

必要的模型权重和对应代码、冻结scale、loss/数据单位/导联定义都包含。
权重复制原始best_r.pt，不转换、不丢EMA/初始化信息；last/best/initial不重复打包。
后续新阶段从P EMA初始化，新建优化器，不能使用--resume P假装继续同一训练。
接收方必须自备同步I和多导联监督数据及划分/QC；本打包不包含原始华为数据、NPZ缓存、
个人metadata或公开数据集。若接收方想精确复现华为验证，应另行获取相同数据/划分/QC。
checkpoint内部自带P的训练身份与public record hashes，不需要发送public NPZ文件。

## 更新打包脚本

本地推送包含脚本的B7-A1分支后，仅在A1目录更新；新增脚本不修改已运行模型包：

```bash
cd /home/qht/huawei_B7_A1
git fetch gitee B7-A1
git merge --ff-only gitee/B7-A1
conda activate BioFlow
```

## 准备交接目录

```bash
cd /home/qht/huawei_B7_A1
handoff_dir="/home/qht/huawei_B7_A1/results/P_handoffs/$(date +%Y%m%d_%H%M%S)"
python -u scripts/prepare_p_handoffs.py --output "$handoff_dir"
```

M2使用已知P路径，T在`/home/qht/huawei_B7_T*`下查找depths=[2,2,6]的已完成P。
找不到或出现不同内容的多个候选会拒绝猜测。此时先找准确路径，再以一个新的输出目录重跑：

```bash
find /home/qht/huawei_B7_T* -type f -path '*/P/best_r.pt'
python -u scripts/prepare_p_handoffs.py --output "$handoff_dir" \
  --t-checkpoint /准确的g6实验目录/P/best_r.pt
```

检查失败时不要继续tar或上传。prepare脚本不训练、不分配GPU、不改源权重、不打包或上传。
输出目录必须不存在，已有结果不覆盖。

每包包含P/best_r.pt、P/config.json、P/completed.json、P/data_contract.json、
对应runtime代码、A10/A40/Q40配置、A1入口、requirements/environment、来源和评分摘要、README、SHA256SUMS。
环境记录仅Python/torch/numpy/scipy版本及torch CUDA build，不导出密钥或整个用户环境。
代码运行时使用B7-A1后端，保持源模型结构，加入huawei10支持；支持新微调，不是复现原P训练环境。

## 分别tar、校验

```bash
(
  set -e
  cd "$handoff_dir/B7_M2_P"
  sha256sum -c SHA256SUMS
  cd "$handoff_dir/B7T_g6_19m_P"
  sha256sum -c SHA256SUMS
  tar -czf "$handoff_dir/B7_M2_P.tar.gz" -C "$handoff_dir" B7_M2_P
  tar -czf "$handoff_dir/B7T_g6_19m_P.tar.gz" -C "$handoff_dir" B7T_g6_19m_P
  cd "$handoff_dir"
  sha256sum B7_M2_P.tar.gz B7T_g6_19m_P.tar.gz > archives.sha256
)
```

## bypy上传

使用已登录可用的bypy，与此前诊断包上传一致：

```bash
remote_dir="/P_handoffs/$(basename "$handoff_dir")"
bypy mkdir "$remote_dir"
bypy upload "$handoff_dir/B7_M2_P.tar.gz" "$remote_dir/B7_M2_P.tar.gz"
bypy upload "$handoff_dir/B7T_g6_19m_P.tar.gz" "$remote_dir/B7T_g6_19m_P.tar.gz"
bypy upload "$handoff_dir/archives.sha256" "$remote_dir/archives.sha256"
bypy list "$remote_dir"
```

bypy远程根是应用目录，不是网盘物理根；按此前的/apps/bypy逻辑查看/下载。
两份tar可分别发给对应队友，archives.sha256是两个归档校验摘要。
如果希望tar文件自身也带时间戳，只需在命名及上传命令中一致替换；远程时间戳目录已避免混淆。

接收方下载后先校验archives.sha256，再解压各自包并在包根执行`sha256sum -c SHA256SUMS`，
根据包内README用--init-checkpoint P/best_r.pt开始微调。
