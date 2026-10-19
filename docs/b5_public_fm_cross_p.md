# PublicFM与当前P慢CNN交叉组合

## 1. 本轮只做什么

不训练，复用已完成的FM预测，固定当前P慢CNN，比较以下三组。每组都评估推理seed42与43，不重新选择训练轮次或挑seed。

| FM | 固定慢CNN | 用途 |
|---|---|---|
| control100 | 当前P最佳 | 当前方案基准 |
| FM-C best100 | 同一个P最佳 | 仅PTB-XL继续预训练后的候选 |
| FM-P best45 | 同一个P最佳 | 加入Chapman公开数据后的候选 |

组合为 `FM输出 − 完整120秒的501点移动平均 + 0.75 × P慢预测`。仍使用Heun16/K16、可靠实际I与目标导联QC，按两个任务平均r比较。只用测试可见的I和人口学预测慢曲线，目标仅参与评分。

当前P完整模型包：

```text
/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt
```

本实验从这个包提取慢CNN。不要传单独训练用的 `P/best.pt`。

## 2. 本地提交后推送，服务器更新现有工作树

本地分支：`B5-best-exp-PublicFM`。本地工作树：

```text
C:\Users\Ashley\Downloads\备赛\B5-CMF工作树\Best-PublicFM
```

先沿用已有方式把本地提交推送到服务器使用的Gitee仓库。服务器的远程名是gitee；本地远程名未必相同，不要直接照搬服务器的远程名。

服务器更新：

```bash
conda activate BioFlow
cd /home/qht/huawei_B5_best_exp_public_fm
git fetch gitee B5-best-exp-PublicFM
git merge --ff-only FETCH_HEAD
sed -i 's/\r$//' scripts/run_b5_public_fm_cross.sh
bash -n scripts/run_b5_public_fm_cross.sh
```

新文件只在scripts/及独立文档中，不更改冻结训练包的代码hash，不需要重新prepare或训练。原始数据、已有权重、study及默认部署模型不会修改。

## 3. GPU1执行六组交叉评估

建议在tmux中执行：

```bash
cd /home/qht/huawei_B5_best_exp_public_fm

bash scripts/run_b5_public_fm_cross.sh --gpu 1
```

默认读取：

```text
PublicFM运行：
/home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1

原seed43预测：
/home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/seed43_review_20261019_125707_2872275

当前P包：
/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt
```

原运行中的 `.npy` 预测缓存必须仍在服务器；报告tar没有这些数组，不能用tar解压目录代替。如果真实seed43目录不同，显式传入：

```bash
bash scripts/run_b5_public_fm_cross.sh \
  --gpu 1 \
  --seed43-review /实际的seed43_review目录
```

运行前核验数据/QC/缓存SHA、相同control100起点、相同归一化及六组采样设置。只检查本轮使用的Huawei数据和预测缓存，不重新读取全部公开波形。P慢CNN只预测一次每条目标，供六组复用。

seed42的control100＋P还会与P/model.pt内同协议参考r比较；超出原包容差则停止，查看 `P_reference_check.json`，避免把基准不一致当作模型收益。不同协议的旧参考只记录不比较。

## 4. 输出及结果判断

新目录自动命名：

```text
/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_日期时间_进程号/
    cross.json
    P_reference_check.json
    summary.csv
    seed_mean.csv
    completed.json
    run.log
    seed42/control100/...
    seed42/FM_C_best100/...
    seed42/FM_P_best45/...
    seed43/...
    bundles/control100__P.pt
    bundles/FM_C_best100__P.pt
    bundles/FM_P_best45__P.pt
```

`summary.csv`包含六组正式r、相对当前control100＋P的增量、相对FM-C＋P的增量，以及新P相对旧慢PCC20的变化。

`seed_mean.csv`为两次分数的算术平均，用于观察稳定性，不代表将两seed波形集成后的成绩。优先看每个seed内部是否同方向改善，再看均值。

三个完整组合包按seed42设置导出，包含各自FM和同一P慢CNN，可后续真实validate验证；本轮不自动替换默认模型。seed43仅评分，不修改部署采样seed。

若报错，保留失败目录和旁边 `.log`，不要覆盖源文件；修复后重新执行会建立新的时间戳目录。

## 5. 打包报告发回

以下自动选最新交叉目录。确认该次运行已完成后执行：

```bash
(
set -euo pipefail
root=/home/qht/huawei/results/B5-best-exp-PublicFM
run=$(find "$root" -maxdepth 1 -type d -name 'cross_P_*' | sort | tail -n 1)
test -n "$run"
test -f "$run/completed.json"
archive="/home/qht/huawei/B5_PublicFM_cross_P_reports_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$archive" --exclude='*.npy' --exclude='*.pt' \
  -C "$(dirname "$run")" "$(basename "$run")"
echo "下载这个压缩包：$archive"
)
```

保留服务器bundles/内三个完整包及原始FM预测；发回报告不需要包含权重与ECG数组。
