# /home/qht/huawei：服务器启动顺序

更新时间：2026-10-09。不要先跑完整公开预训练；按代码同步、缓存上传、环境/数据预检、1窗/4窗过拟合、正式训练推进。

## 1. 确认拿到代码

本地分支未推送时，服务器只有同名分支并不表示已有本次新增源码。先在本地完成代码提交并执行 git push origin baseline/B5；再在服务器执行：

```bash
cd /home/qht/huawei
git status --short --branch
git fetch origin
git switch baseline/B5
git pull --ff-only origin baseline/B5
test -f baselines/B5/README.md
test -f baselines/B5/overfit.py
```

服务器有未提交修改时先保留自己的修改，不执行 reset --hard。也可直接传输本次完整 B5 源码和配置，但 Git 同步更容易核对版本。

## 2. 传哪些数据

默认配置的 sibling 布局：

```text
/home/qht/
  huawei/                         Git仓库
  task1_output_v2/                 整个文件夹
  task2_output/                    整个文件夹
  Data/
    userinfobean.csv
    ptbxl_database.csv
    records500/                   完整 .hea/.dat 配对
```

必须传最新 task1_output_v2，而不是旧 task1_output。首版严格同步 train-only 索引也引用 task2_train_target.npy，验证加载 Task2 context/target，因此只传 Task1 不够；Task2 整个文件夹也要传。

task1_output_v2 内保留 train/validation input/target NPY、context_valid_mask NPY、window_metadata CSV 及配套文件，不重新排序窗口。task2_output 保留全部 train/validation input/target NPY 与 metadata。

metadata/subject_split.csv、d12_strict_pretrain_index.csv、device_interpretation_qc.csv、pair_manifest 等已在仓库内，应与缓存版本匹配。userinfobean.csv 在 Data 下，Git 不会自动带上它。

首版 B5-U 直接使用缓存中目标同窗 I，没有历史 ECG/PPG 输入，不必为首版传所有手表/PPG zip、原始 XML 或体脂秤原始文件；以后重建缓存或加历史条件时再需要原始数据。

公开集可以随后上传或直接在服务器下载；1窗/4窗诊断不需要公开集。正式公开预训练需要 ptbxl_database.csv 与完整 records500，records100 不能替代。

若数据已经放在 /home/qht/huawei 内，则编辑 configs/experiments/b5_base.yaml 的 paths：

```yaml
paths:
  data_root: Data
  huawei_data_root: .
  ptbxl_root: ${data}
```

此时缓存目录应为 /home/qht/huawei/task1_output_v2 和 /home/qht/huawei/task2_output。选择一种布局，不能混用。Windows 的原始 source_path 元数据不需要批量替换；首版从 NPY 缓存读取，不按这些字段重新打开 XML。

## 3. 环境和无训练检查

```bash
cd /home/qht/huawei
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -r baselines/B5/requirements.txt
python -m unittest baselines.B5.tests.test_b5 -v
python -m baselines.B5.prepare --config configs/experiments/b5_local_meta.yaml --fit-scales --report results/B5/preflight_huawei.json
```

使用自己的 Python/conda 环境；已有可用 CUDA PyTorch 时保持正确版本。requirements 不负责选择 NVIDIA 驱动/CUDA 安装方式。CUDA不可用先修环境，不用CPU慢跑全量预训练。

已有同协议冻结尺度文件时去掉 --fit-scales，直接复用，不覆盖重拟合。prepare不训练，也不下载。

## 4. 小集合诊断

```bash
python -m baselines.B5.overfit --config configs/experiments/b5_local_meta.yaml --windows 1 --steps 1000 --device cuda --output-dir results/B5/overfit_w1 --execute-training
python -m baselines.B5.overfit --config configs/experiments/b5_local_meta.yaml --windows 4 --steps 1000 --device cuda --output-dir results/B5/overfit_w4 --execute-training
```

先运行第一条，看真实ODE生成能否靠近训练target，再运行第二条。观察诊断JSONL中的loss趋势、训练窗口r与缩放空间RMSE，必要时叠图比较。这里的指标是训练样本诊断，不是正式验证结果。代码不输出正式 checkpoint，避免将诊断模型直接当成预训练模型。

本次开发没有运行上述优化命令。是否执行由服务器用户显式决定。

## 5. 正式实验

华为E1从零基准可在公开下载未完成时先运行；公共E2需要完整性检查通过：

```bash
python -m baselines.B5.train_huawei --config configs/experiments/b5_local_meta.yaml --device cuda --execute-training
python -m baselines.B5.prepare --config configs/experiments/b5_public_meta.yaml --check-public --report results/B5/preflight_public.json
python -m baselines.B5.train_public --config configs/experiments/b5_public_meta.yaml --device cuda --execute-training
python -m baselines.B5.train_huawei --config configs/experiments/b5_finetune_meta.yaml --init-checkpoint results/B5/E2_public_meta/best.pt --device cuda --execute-training
```

不要让无 checkpoint 的 finetune 配置静默从随机初始化训练；主线会明确拒绝。E1与E2在同一华为验证集比较。人口学I-only对照、后续历史模态和骨干实验见 baselines/B5/README.md。
