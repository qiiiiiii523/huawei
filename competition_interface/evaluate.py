# -*- coding: utf-8 -*-
"""赛题 1 评分参考程序，仅用于理解评分方法和检查结果格式。

用法：
    python evaluate.py --pred-dir results --gold-dir aligned_gold

pred-dir 中的预测 CSV 必须由同目录 output.py 生成；gold-dir 直接指向 process
生成的 d12 CSV 目录，每人一份金标准，三个任务共用同一份金标准。金标准视为
已经处理完成，程序只读取、不修改。参考程序不提供金标准，不代表主办方正式
评测环境；最终成绩以主办方评测结果为准。

计分方式：三个任务均不使用 I 导联。先对 II、III、aVR、aVL、aVF、V1-V6
逐导联计算 Pearson 相关系数，再对导联和人员取算术平均。负相关保留负号，
恒定信号相关系数记为 0。最终参考分数为：

    score = 0.50 * R_gen1 + 0.25 * R_gen6_scale + 0.25 * R_gen6_d6

此外，gen6_scale 和 gen6_d6 分别计算 V1-V6 六个胸导联的 RMSE，单位为 uV。
评测报告记录每个胸导联、每个六导联任务以及两个六导联任务合并后的平均 RMSE。
RMSE 只用于观察绝对误差，不参与上面的相关系数总分。命令行仅打印汇总结果。

gold-dir 中每人一份 CSV，文件名为被试编号，列为 userid 和固定 12 导联。
本程序只输出汇总 JSON，不写入或修改预测和金标准文件。

依赖：numpy。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from output import LEADS, TASKS, read_result


WEIGHTS = {'gen1': 0.50, 'gen6_scale': 0.25, 'gen6_d6': 0.25}
CORRELATION_LEADS = tuple(lead for lead in LEADS if lead != 'I')
RMSE_LEADS = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')


class EvaluationError(ValueError):
    """预测或参考金标准不满足评测格式。"""


def signal_metrics(prediction, gold, *, include_rmse=False):
    """按原评测逻辑计算 Pearson；需要时同时计算原始 uV 电压上的 RMSE。"""
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(gold, dtype=np.float64)
    if pred.ndim != 1 or true.ndim != 1 or pred.shape != true.shape or pred.size < 2:
        raise EvaluationError('信号须为等长一维数组，且至少包含两个采样点')
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise EvaluationError('信号包含 NaN 或无穷值')

    # 与正式评测代码一致：缩放后中心化，避免极大有限值在相关系数中溢出。
    x = pred / (float(np.max(np.abs(pred))) or 1.0)
    y = true / (float(np.max(np.abs(true))) or 1.0)
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if np.all(pred == pred[0]) or np.all(true == true[0]) or nx == 0 or ny == 0:
        correlation = 0.0
    else:
        correlation = float(np.clip(np.dot(x / nx, y / ny), -1.0, 1.0))

    result = {'correlation': correlation}
    if include_rmse:
        # 与原代码相同：在原始 uV 数值上计算 sqrt(mean((pred-true)^2))。
        # 先按最大绝对误差缩放，只是为了避免平方溢出，不会改变 RMSE 数值。
        with np.errstate(over='ignore', invalid='ignore'):
            difference = pred - true
            scale = float(np.max(np.abs(difference)))
            value = scale * float(np.sqrt(np.mean((difference / scale) ** 2))) if scale else 0.0
        if not np.isfinite(value):
            raise EvaluationError('RMSE 超出 float64 可表示范围')
        result['rmse_uv'] = value
    return result


def pearson(prediction, gold):
    """兼容简化版原有调用，返回 signal_metrics() 中的 Pearson 相关系数。"""
    return signal_metrics(prediction, gold)['correlation']


def read_gold(path):
    """直接读取 process 生成的 d12 CSV。"""
    values = {lead: [] for lead in LEADS}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            for lead in LEADS:
                values[lead].append(float(row[lead]))
    return {lead: np.asarray(values[lead], dtype=np.float64) for lead in LEADS}


def evaluate_subject(result, gold):
    """完全按原评测层级计算一位被试的相关系数及 V1-V6 RMSE。"""
    tasks = {}
    for task in TASKS:
        leads = result[task]['leads']
        if set(leads) != set(LEADS):
            raise EvaluationError('%s 必须恰好包含 12 导联' % task)
        lead_metrics = {}
        for lead in CORRELATION_LEADS:
            lead_metrics[lead] = signal_metrics(
                leads[lead], gold[lead],
                include_rmse=task != 'gen1' and lead in RMSE_LEADS)
        tasks[task] = {
            'mean_correlation': float(np.mean([
                metrics['correlation'] for metrics in lead_metrics.values()
            ])),
            'leads': lead_metrics,
        }
        if task != 'gen1':
            tasks[task]['mean_rmse_uv'] = float(np.mean([
                lead_metrics[lead]['rmse_uv'] for lead in RMSE_LEADS
            ]))
    score = sum(WEIGHTS[task] * tasks[task]['mean_correlation'] for task in TASKS)
    return {'userid': result['userid'], 'score': score, 'tasks': tasks}


def summarize(subjects):
    """按原代码顺序汇总：导联→人员→两种六导联来源，全部算术平均。"""
    if not subjects:
        raise EvaluationError('没有可评测的人员')
    tasks = {}
    for task in TASKS:
        tasks[task] = {
            'mean_correlation': float(np.mean([
                subject['tasks'][task]['mean_correlation'] for subject in subjects
            ]))
        }
        if task != 'gen1':
            tasks[task]['mean_rmse_uv'] = float(np.mean([
                subject['tasks'][task]['mean_rmse_uv'] for subject in subjects
            ]))
    gen6 = {
        'mean_correlation': (
            tasks['gen6_scale']['mean_correlation'] +
            tasks['gen6_d6']['mean_correlation']) / 2,
        'mean_rmse_uv': (
            tasks['gen6_scale']['mean_rmse_uv'] +
            tasks['gen6_d6']['mean_rmse_uv']) / 2,
    }
    return {
        'score': sum(WEIGHTS[task] * tasks[task]['mean_correlation'] for task in TASKS),
        'subject_count': len(subjects),
        'tasks': tasks,
        'gen6': gen6,
    }


def evaluate_directory(pred_dir, gold_dir):
    """按 CSV 文件名匹配人员，按人员等权汇总并计算参考总分。"""
    predictions = {path.stem: path for path in Path(pred_dir).glob('*.csv')}
    gold_files = {path.stem: path for path in Path(gold_dir).glob('*.csv')}
    if not predictions:
        raise EvaluationError('预测目录没有 CSV')
    if set(predictions) != set(gold_files):
        missing = sorted(set(gold_files) - set(predictions))
        extra = sorted(set(predictions) - set(gold_files))
        raise EvaluationError('人员不一致；预测缺少: %s；预测多出: %s' %
                              (', '.join(missing) or '无',
                               ', '.join(extra) or '无'))
    subjects = []
    for userid in sorted(predictions):
        try:
            result = read_result(predictions[userid], expected_userid=userid)
            gold = read_gold(gold_files[userid])
            subjects.append(evaluate_subject(result, gold))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EvaluationError('%s: %s' % (userid, exc)) from exc
    return {
        'reference_only': True,
        'metric': 'mean_pearson_correlation',
        'weights': WEIGHTS.copy(),
        'correlation_leads': list(CORRELATION_LEADS),
        'rmse_leads': list(RMSE_LEADS),
        'rmse_aggregation': 'arithmetic_mean_over_V1_to_V6_subjects_and_two_gen6_sources',
        'summary': summarize(subjects),
        'subjects': subjects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-dir', type=Path, required=True,
                        help='output.py 生成的预测 CSV 目录')
    parser.add_argument('--gold-dir', type=Path, required=True,
                        help='处理后的 d12 金标准 CSV 目录')
    args = parser.parse_args()
    try:
        report = evaluate_directory(args.pred_dir, args.gold_dir)
        print(json.dumps(report['summary'], ensure_ascii=False, indent=2, allow_nan=False))
    except (EvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
