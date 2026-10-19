# -*- coding: utf-8 -*-
"""赛题 1 输出接口：每位被试生成一份包含三个任务的固定格式 CSV。

三个任务必须同时提交：

    gen1        单导联生成 12 导联
    gen6_scale  体脂秤 6 导联生成 12 导联
    gen6_d6     心电图机 d6 生成 12 导联

每个任务的调用数据均为：

    {
      'fs': 500,
      'leads': {
        'I': [...], 'II': [...], 'III': [...],
        'aVR': [...], 'aVL': [...], 'aVF': [...],
        'V1': [...], 'V2': [...], 'V3': [...],
        'V4': [...], 'V5': [...], 'V6': [...]
      }
    }

12 个导联都必须提供，每个导联恰好 60000 点，对应 120 秒、500 Hz，单位为 uV。
值可为一维列表、array 或 NumPy 一维数组，但必须全部是有限数值。接口只验证和
保存，不会截断、补齐、滤波、重采样或对齐；fs=500 是调用者对真实采样率的声明。

每位被试调用一次：

    from output import save_result

    gen1 = {'fs': 500, 'leads': leads_from_single}
    gen6_scale = {'fs': 500, 'leads': leads_from_scale}
    gen6_d6 = {'fs': 500, 'leads': leads_from_d6}
    save_result('HNU00001', gen1, gen6_scale, gen6_d6,
                output_dir='results')

生成 results/HNU00001.csv。CSV 固定为 39 列：userid、sample_index、fs，
以及带 gen1_、gen6_scale_、gen6_d6_ 前缀的各 12 导联。sample_index 为
0 至 59999，第 k 点对应 k/500 秒。save_result 默认禁止覆盖已有结果；新一轮
运行应先使用空结果目录，或确认后传 overwrite=True。validate_result() 只校验
调用数据而不写文件；read_result() 读取并校验已经生成的 CSV。

Python 3.9+，仅依赖标准库。
"""

import csv
import math
import os
import re
from array import array
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

__all__ = ['LEADS', 'TASKS', 'FS', 'N_SAMPLES', 'OutputFormatError',
           'validate_result', 'save_result', 'read_result']

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
TASKS = ('gen1', 'gen6_scale', 'gen6_d6')
FS = 500
N_SAMPLES = 120 * FS
_HEADER = ('userid', 'sample_index', 'fs') + tuple(
    '%s_%s' % (task, lead) for task in TASKS for lead in LEADS)




class OutputFormatError(ValueError):
    """任务缺失、导联缺失、采样率或采样点数不符、数值无效。"""


def _userid(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise OutputFormatError('userid 必须是由字母、数字、下划线或连字符组成的非空编号')
    return value


def _finite(value):
    if isinstance(value, (str, bytes, bool)):
        raise ValueError('信号必须是数值，不能是字符串或布尔值')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('信号不能含 NaN 或无穷值')
    return number


def _task_data(name, payload):
    if not isinstance(payload, Mapping) or 'fs' not in payload or 'leads' not in payload:
        raise OutputFormatError('%s 必须包含 fs 和 leads 两个字段' % name)
    try:
        fs = _finite(payload['fs'])
    except (ValueError, TypeError, OverflowError) as exc:
        raise OutputFormatError('%s 的 fs 必须是数值 500' % name) from exc
    if fs != FS:
        raise OutputFormatError('%s 采样率必须为 500 Hz，收到 %s' % (name, fs))
    if payload.get('unit', 'uV') not in ('uV', 'µV', 'μV'):
        raise OutputFormatError('%s 电压单位必须为 uV' % name)
    leads = payload['leads']
    if not isinstance(leads, Mapping) or set(leads) != set(LEADS):
        missing = [lead for lead in LEADS if not isinstance(leads, Mapping) or lead not in leads]
        raise OutputFormatError('%s 必须恰好包含 12 导联；缺失: %s' % (name, ', '.join(missing) or '无（存在多余字段）'))
    normalized = {}
    for lead in LEADS:
        signal = leads[lead]
        if isinstance(signal, (str, bytes, Mapping)) or getattr(signal, 'ndim', 1) != 1:
            raise OutputFormatError('%s_%s 必须是一维数值序列' % (name, lead))
        try:
            if len(signal) != N_SAMPLES:
                raise OutputFormatError('%s_%s 必须有 60000 点，收到 %s' % (name, lead, len(signal)))
            values = array('d')
            for index, value in enumerate(signal):
                try:
                    values.append(_finite(value))
                except (ValueError, TypeError, OverflowError) as exc:
                    raise OutputFormatError('%s_%s 第 %d 点无效: %s' % (name, lead, index, exc)) from exc
            # 防止自定义序列长度声明与实际迭代结果不一致。
            if len(values) != N_SAMPLES:
                raise OutputFormatError('%s_%s 实际采样点数不是 60000' % (name, lead))
        except (TypeError, OverflowError) as exc:
            raise OutputFormatError('%s_%s 必须是一维数值序列' % (name, lead)) from exc
        normalized[lead] = values
    return dict(fs=FS, unit='uV', leads=normalized)


def validate_result(userid, gen1, gen6_scale, gen6_d6):
    """检查编号及三个任务，返回标准化数据；不写文件、不计算评分。"""
    return dict(userid=_userid(userid), gen1=_task_data('gen1', gen1),
                gen6_scale=_task_data('gen6_scale', gen6_scale),
                gen6_d6=_task_data('gen6_d6', gen6_d6))


def save_result(userid, gen1, gen6_scale, gen6_d6, output_dir='results', *, overwrite=False):
    """保存 output_dir/userid.csv，返回 Path；默认禁止覆盖。

    gen1、gen6_scale、gen6_d6 均须为 {'fs': 500, 'leads': {12个导联: 一维信号}}。
    可选 unit 字段必须为 uV。三个任务均必填；缺少调用参数会抛 TypeError。
    默认已有同名文件时抛 FileExistsError；overwrite=True 时原子替换该文件。
    """
    result = validate_result(userid, gen1, gen6_scale, gen6_d6)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (userid + '.csv')
    if target.exists() and not overwrite:
        raise FileExistsError('输出已存在: %s' % target)
    temporary = directory / ('.%s.%s.tmp' % (userid, uuid4().hex))
    signals = [result[task]['leads'][lead] for task in TASKS for lead in LEADS]
    try:
        with temporary.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(_HEADER)
            for index in range(N_SAMPLES):
                writer.writerow([userid, index, FS] + [signal[index] for signal in signals])
        if overwrite:
            os.replace(temporary, target)
        else:
            # 链接目标创建是独占操作，避免并发时覆盖另一份结果。
            os.link(temporary, target)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


def read_result(path, expected_userid=None):
    """读取并校验 CSV；返回 userid、gen1、gen6_scale、gen6_d6。

    返回的每个任务包含 fs=500、unit='uV' 和 12 导联 array('d')。
    expected_userid 可用于检查该文件是否属于预期用户。此函数供自检和参考评测使用。
    """
    if expected_userid is not None:
        _userid(expected_userid)
    result = {task: dict(fs=FS, unit='uV', leads={lead: array('d') for lead in LEADS})
              for task in TASKS}
    userid = None
    count = 0
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header != list(_HEADER):
                raise OutputFormatError('CSV 表头不符，必须包含 gen1、gen6_scale、gen6_d6 各 12 导联及固定元数据列')
            for row in reader:
                if len(row) != len(_HEADER) or count >= N_SAMPLES:
                    raise OutputFormatError('CSV 第 %d 行列数不符或数据超过 60000 行' % reader.line_num)
                current_id = _userid(row[0])
                if userid is None:
                    userid = current_id
                if current_id != userid or (expected_userid is not None and current_id != expected_userid):
                    raise OutputFormatError('CSV 用户编号不一致或与预期用户不符')
                try:
                    if int(row[1]) != count:
                        raise ValueError('sample_index 必须依次为 0 至 59999')
                    if float(row[2]) != FS:
                        raise ValueError('fs 必须为 500 Hz')
                    for position, (task, lead) in enumerate(
                            ((task, lead) for task in TASKS for lead in LEADS), start=3):
                        value = float(row[position])
                        if not math.isfinite(value):
                            raise ValueError('%s_%s 包含非有限数值' % (task, lead))
                        result[task]['leads'][lead].append(value)
                except (ValueError, OverflowError) as exc:
                    raise OutputFormatError('CSV 第 %d 行: %s' % (reader.line_num, exc)) from exc
                count += 1
    except (UnicodeError, csv.Error) as exc:
        raise OutputFormatError('CSV 编码或格式错误') from exc
    if count != N_SAMPLES:
        raise OutputFormatError('CSV 必须有 60000 行数据，收到 %d' % count)
    return dict(userid=userid, **result)
