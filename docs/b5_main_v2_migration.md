# B5 对齐新数据和新版main

核对基准为本地main提交 `6dcb01c`，实际代码优先于提交较早的《新版main分支完整实现说明》中的版本文字。

## 本次范围

保留B5-U网络、人口学条件和FM训练目标。当前best.pt改按50分主分＋10分加分的总分选择，并独立保存best_r.pt、best_bonus.pt和last.pt；见[评分与保存说明](b5_score60_checkpoint_policy.md)。没有新增context编码器、T或F模块。
同步I与target保持raw电压，只除冻结scale，输出仅乘一次scale。缓存中的独立context不按target窗口序号取窗，也不作为网络输入。
训练使用新版strict去重的1872个窗口/156条物理target，排除不可靠I并按导联质量mask监督；不把两个Task的重复配对当作新增训练目标。
验证覆盖Task1的27配对/324窗，以及Task2机器40配对/480窗和体脂秤25配对/300窗。每个配对固定12窗/120秒。

Task2必须保留input_type，两组分别算r和V1–V6 RMSE，设备间固定1:1；总分调用新版四参数接口。平均RMSE只是诊断值，不用它再换算加分。
旧的0.66949等分数来自旧验证集/旧设备范围，不能与新版结果直接作模型提升比较。要比较模型，全部在同一新版验证集和同一采样设置重评。

## 服务器需要的新文件

```text
/home/qht/
├── task1_record_context_v3/        # 完整新版缓存，包含context/mask/索引/target
├── task2_record_context_v3/        # 完整新版缓存，包含两设备
└── huawei/
    ├── metadata/                  # 跟随本分支更新的split/pair/QC/strict
    │   └── userinfobean_utf8.csv   # 需单独上传，人口学文件不进Git
    ├── data/userinfobean.csv       # 可替代上面的UTF8副本，须为更新后的文件
    └── results/B5/
        ├── shared/preprocessing_scales.npz
        └── E2_finetune_meta/best.pt
```

两个新版缓存目录完整上传，不能把旧task1_output_v2/task2_output改名当成新缓存。它们的target/context索引互相独立，数组顺序也改变，必须与Git中的新版strict/配对表对应。
本基准不使用body B overlay，不需为B5额外上传体脂秤B滤波缓存或公开原始数据。正式两设备评分仍需要A缓存中的全部机器、体脂秤验证配对。

人口学优先使用配置指定的 `metadata/userinfobean_utf8.csv`；不存在时读 `data/userinfobean.csv`，支持原GB18030编码，预检会报告实际路径。应上传新版185行文件，不沿用旧CSV。也可在YAML明确写绝对路径。

## 先预检，不训练

用户提交本地合并及适配代码并推送B5后，服务器：

```bash
cd /home/qht/huawei
git fetch origin
git switch B5
git pull --ff-only origin B5
python -m unittest discover -s baselines/B5/tests -v

python -m baselines.B5.prepare \
  --config configs/experiments/b5_finetune_meta.yaml \
  --report results/B5/main_v2_preflight.json
```

ready=true才表示读取通过。核对1872训练窗和324/780验证窗，Task2设备配对40/25。检查最新版数据可能耗时，但不启动训练。
若数据在另一目录，修改 `paths.huawei_data_root`；人口学放在仓库data时 `paths.data_root: data`。

## 保持旧权重的数值尺度

初始化已有raw-target checkpoint时，继续使用对应旧冻结scale，不因为数据增加就重新fit/替换；训练会校验scale sha一致。
已有冻结scale只含d12和ecg_machine_i即可，因为B5不编码context。main预处理协议的raw-target语义未变，旧权重的架构hash也未变。
若另拟合新scale，不能直接加载旧权重训练而忽略数值表示变化；本分支没有跨scale迁移模块。
`prepare --fit-scales`仅供没有预训练权重的新尺度准备，不能覆盖已有文件。新版拟合只统计strict train中该导联质量可靠的窗口；不统计validation或context填充值。

## 先重评旧最佳模型

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_finetune_meta.yaml \
  --checkpoint results/B5/E2_finetune_meta/best.pt \
  --device cuda \
  --output-dir results/B5/main_v2_old_best_evaluation
```

默认16步、K16；目录须为空。报告包含两设备r、两设备胸导联RMSE、等权加分和总分。输出与旧报告分开。

## 在新数据上微调

默认输出为新目录 `results/B5/main_v2_score60_finetune_meta`，不会覆盖旧E2目录：

```bash
set -o pipefail
mkdir -p results/B5/main_v2_score60_finetune_meta
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_finetune_meta.yaml \
  --init-checkpoint results/B5/E2_finetune_meta/best.pt \
  --device cuda --execute-training \
  2>&1 | tee results/B5/main_v2_score60_finetune_meta/train.log
```

只加载EMA模型权重；优化器、更新次数、数据manifest和best_score重新建立，不把旧最佳分数用作新版阈值。
保留原100轮、每5轮验证和期间K1的配置；最终评估仍K16。若另设K16期间选模，须在新运行开始前改配置；不能更改采样后直接resume。

新运行中断后可续训本组last：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.train_huawei \
  --config configs/experiments/b5_finetune_meta.yaml \
  --resume results/B5/main_v2_score60_finetune_meta/last.pt \
  --device cuda --execute-training
```

旧数据或旧选模规则last.pt的resume会报协议变化；应使用init-checkpoint开启新运行。新版resume检查实际缓存字节、split/pair/QC/人口学/strict索引、模型、损失及验证采样，防止同目录换数据后继续旧历史。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.validate \
  --config configs/experiments/b5_finetune_meta.yaml \
  --checkpoint results/B5/main_v2_score60_finetune_meta/best.pt \
  --device cuda \
  --output-dir results/B5/main_v2_score60_finetune_meta/final_evaluation
```

## 测试推理

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.B5.predict \
  --checkpoint results/B5/main_v2_score60_finetune_meta/best.pt \
  --anchor /path/to/visible_synchronous_I.npy \
  --metadata-json /path/to/observed_demographics.json \
  --record-id test_record_001 \
  --input-fs 500 --input-unit uV \
  --device cuda \
  --output results/B5/test_predictions/test_record_001.npy
```

人口学可省略并按缺失mask处理。正式入口要求真实120秒同步I，长输入只取前120秒，短输入报错，不padding隐藏目标。输出默认保持输入采样率、固定120秒、12导联、μV。
`--allow-variable-length`保留旧任意长度诊断能力，但不能当作新版正式120秒提交流程。predict不读取真实target，也不依赖context；JSON明确记录context_used=false及目标时长。

本机没有训练。CPU功能检查与真实缓存只读检查不等于精度验证；GPU性能和新版数据成绩需在服务器确认。
