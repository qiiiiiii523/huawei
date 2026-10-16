# B5-best默认采用beta=0.75

新模型包保存`slow_beta=0.75`。最终缺失11导联为：

```text
FM预测 − 对完整120秒FM预测做501点居中移动平均
       ＋ 0.75 × S50-F预测慢曲线
```

I输出仍来自原FM的I头。两个源网络、权重、尺度、条件和训练损失不变，不需要重新训练。Heun16/K16/seed42不变。

## 已有验证依据

| 推理seed | beta | Task1 r | Task2 r | 平均r |
|---|---|---|---|---|
| 42 | 0.5 | 0.691914 | 0.685830 | 0.688872 |
| 42 | 0.75 | 0.698164 | 0.691673 | 0.694918 |
| 42 | 1 | 0.695855 | 0.689403 | 0.692629 |
| 43 | 0.75 | 0.698445 | 0.691056 | 0.694750 |
| 43 | 1 | 0.696160 | 0.688680 | 0.692420 |

0.75是已测试候选中更好的选择；两次噪声复核仍来自同一批验证患者，不保证新患者同样获益。RMSE不参与当前选模。

## 服务器更新后，导出并验证新模型包

在BioFlow环境执行。已有worktree仍使用原位置，不切换其他训练目录：

```bash
cd /home/qht/huawei_B5_best
git fetch gitee B5-best
git switch --detach FETCH_HEAD
bash scripts/run_b5_best.sh --gpu 1
```

脚本按顺序导出、评估，无训练；默认结果：

```text
/home/qht/huawei/results/B5-best/stable_beta075_v2/
    model.pt
    model.json
    evaluation/evaluation.json
    evaluation/correlation_score.csv
```

成功时evaluation.json中`completed=true`、`reference_matched=true`、`slow_beta=0.75`；平均r应接近0.69491841，每任务参考容差2e-5。若目录已存在，使用`--root`指定新目录，勿覆盖原包。真实源权重和华为数据需在服务器保留；本机只做CPU合成检查。

之后predict/validate指向这个新model.pt即可，不用额外传beta。新记录输入示例见`docs/b5_best.md`。

## 旧模型包与历史诊断

- v1包没有slow_beta字段，更新代码后仍按beta=1推理，旧参考分数保持其原含义。
- v2包必须显式保存合法的slow_beta；新默认配置为0.75，错误或缺失参数会报错。
- 只更新配置不会改变已导出的旧包，必须重新导出，无需重新训练源权重。
- 历史诊断读取的原组合数组仍是beta=1，使用单独保留的`reference_qc_r_beta1`核验，不能拿0.75的新参考比较。
- 下一轮训练实验中，对照和实验采用相同beta，避免把组合系数变化当作模块收益。
