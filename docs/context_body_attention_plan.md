# 体脂秤 context 局部注意力编码器试验

## 目的

检查保留窗口内局部时间结构的 context 编码器，能否让正确配对的体脂秤 ECG 对 B5 11 导联预测产生可重复收益。

## 只改编码器

- B5 主干继续使用 results/B5/context_start/best.pt，冻结不训练。
- 继续使用 A_frozen_meanstd_v1/best.pt 的已训练 context CNN、投影层、记录级均值/标准差融合层和 FiLM 参数。
- 新的时间注意力层权重和偏置置零。注意力 logits 加上 log(valid_support)，因此初始池化权重与原有 support 加权时间均值相同，试验从已有模型功能附近开始。
- 训练时只更新 body_encoder.*。FiLM、B5 主干、预处理、训练数据、损失、随机种子和 Heun16/K16 验证协议保持固定。
- 独立输出到 results/context_body/A_frozen_attention_v1/，不覆盖旧实验。

该结构以可学习的时间权重池化每个窗口 CNN 特征，仍保留原有的跨窗口加权均值与标准差。它不硬编码 R 峰，也不声称做了逐拍对齐。后续可以根据结果决定是否需要显式 beat-level 输入。

## 运行

在远端先进入项目根目录并启用 GPU 环境：

    cd /root/autodl-tmp/Huawei_B5_context/huawei
    source ../.venv_gpu/bin/activate
    export PYTHONPATH=.

先检查远端已存在：

- results/B5/context_start/best.pt
- results/context_body/A_frozen_meanstd_v1/best.pt
- 体脂秤 Task 2 数据及 demographic 输入文件

GPU 预检：

    python scripts/train_b5_body_frozen_adapter.py --encoder-variant attention --device cuda --preflight

训练：

    python scripts/train_b5_body_frozen_adapter.py --encoder-variant attention --device cuda --epochs 10 --batch-size 4 --accumulation 2 --learning-rate 1e-4 --validate-every 5

训练完成后，用同一个新 checkpoint 运行四种推理条件：

    python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_attention --mode A_true --device cuda
    python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_attention --mode A_disabled --device cuda
    python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_attention --mode A_shuffled --shuffle-seed 101 --device cuda
    python scripts/evaluate_b5_body_diagnostics.py --checkpoint-kind frozen_attention --mode A_shuffled --shuffle-seed 202 --device cuda

判读以同一模型内正确配对是否稳定优于跨患者打乱为主，同时报告禁用 context、r_missing11、V1–V6 RMSE 和逐配对误差。仅真实配对同时优于打乱和禁用，才作为保留该结构的支持证据。单次验证集选模仍属探索性结果。
