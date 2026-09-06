# AutoDL 运行说明

图3的 vGPU-32GB、16核CPU、80GB内存足够当前 B1。网络约16.3万参数；32GB显存不是瓶颈。
训练脚本支持 `--device auto`，有 CUDA 时自动使用 GPU，否则回退 CPU；`--device cuda` 可强制检查 CUDA。

## 上传范围

从一个干净 clone 开始，只上传仓库代码和必要数据目录：`task1_output`、`task2_output`、
`task2_body_scale_ablation`、`task1_rpeak_pseudo_output`、`task1_rpeak_pseudo_output_c2`。
不要上传本机 `results/`、checkpoint、预测 NPY、`task2_project/.venv` 或原始 ECG 压缩包（除非数据协议明确允许）。
当前本地副本因已有结果约28.6GB；新机器不应直接打包整个副本。

## 运行检查

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m b1.preflight
python -m b1.train_task1_a --loss-variant L0 --epochs 100 --device auto --torch-threads 8
python -m b1.train_task1_a --loss-variant L1 --epochs 100 --device auto --torch-threads 8
```

保持 seed、batch、AdamW、学习率、weight decay、epoch 和评估规则不变；GPU 只改变执行设备，不改变实验定义。
