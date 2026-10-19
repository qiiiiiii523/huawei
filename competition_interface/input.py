# -*- coding: utf-8 -*-
"""赛题 1 输入接口（Python 3.9+，仅依赖标准库）。

参赛者通常只需导入本文件，不应自行解析底层 TXT、CSV 或 XML。默认目录结构：

    test_data/
      userinfo.txt
      watch/HNU00001/*.txt
      scale/*.csv
      Electrocardiograph/d6_test/*.xml
      Electrocardiograph/d12_test/*.xml

推荐逐人读取，避免一次把全部波形载入内存：

    import input as competition_input

    competition_input.configure('./test_data')
    for person in competition_input.iter_samples():
        userid = person['subject_id']
        info = person['userinfo']
        watch_records = person['watch']
        scale_records = person['scale']
        d6_records = person['ele']['d6']
        d12_records = person['ele']['d12']

一位被试的固定返回结构：

    {
      'subject_id': 'HNU00001',
      'userinfo': {
        'externalid': 'HNU00001', 'height': 170.0,
        'weight': 65.0, 'age': 30.0, 'gender': '男'
      },
      'watch': [{'record_id': '...', 'leads': {'voltage': [...]}}],
      'scale': [{'record_id': '...', 'leads': {'1': [...], ..., '12': [...]}}],
      'ele': {
        'd6': [{'record_id': '...', 'leads': {
          'I': [...], 'II': [...], 'III': [...],
          'aVR': [...], 'aVL': [...], 'aVF': [...]
        }}],
        'd12': [{'record_id': '...', 'leads': {'I': [...]}}]
      }
    }

每条记录只有 record_id 和 leads。leads 的值是按时间排列的 Python 数值列表，
电压统一为 uV。d6 只返回 6 个肢体导联，d12 只返回 I 导联；scale 保留原列名
"1" 至 "12"，watch 的通道名为 voltage。不同设备和多次采集分别返回，不自动
同步、截断、补齐、滤波或重采样。使用列表第 0 项前应先判断列表是否为空。

read_subject(id) 读取一人；iter_samples() 逐人读取；read_all() 返回
{编号: 一人数据}，但可能占用较多内存。缺少某设备时返回空列表，文件损坏时抛出
DataFormatError。默认用户信息为 test_data/userinfo.txt；也可用 configure()、
COMPETITION_DATA_ROOT 和 COMPETITION_USERINFO 指定位置。导入模块本身不会读取数据。
"""

import csv
import io
import json
import math
import os
from pathlib import Path
from xml.etree import ElementTree as ET

__all__ = [
    'DataFormatError', 'Dataset', 'configure', 'list_subjects', 'read_userinfo',
    'read_subject', 'read_all', 'iter_samples', 'read_watch', 'read_scale', 'read_ele',
]

LIMB_LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
_INFO_FIELDS = ('externalid', 'height', 'weight', 'age', 'gender')
_NS = {'h': 'urn:hl7-org:v3'}
_LEAD_CODES = dict(zip(
    ('MDC_ECG_LEAD_' + name for name in ('I', 'II', 'III', 'AVR', 'AVL', 'AVF')),
    LIMB_LEADS,
))


class DataFormatError(ValueError):
    """输入文件格式错误或缺少必需字段。"""


def _text(path):
    raw = path.read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise DataFormatError('%s: 无法解码文本' % path.name)


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('数据包含非有限数值')
    return number


def _unit_factor(unit):
    unit = unit.strip().replace('μ', 'u').replace('µ', 'u')
    factors = {'uV': 1.0, 'mV': 1000.0, 'V': 1000000.0}
    if unit not in factors:
        raise ValueError('不支持的电压单位: %s' % unit)
    return factors[unit]


def _record(path, leads):
    return dict(record_id=path.stem, leads=leads)


def _watch(path, subject):
    values = []
    for line in _text(path).splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        voltage = frame['voltage']
        if not voltage:
            raise ValueError('watch 存在空波形帧')
        values.extend(_number(item['value']) * _unit_factor(item['unit'])
                      for item in voltage)
    if not values:
        raise ValueError('watch 文件没有波形数据')
    return _record(path, {'voltage': values})


def _scale_header(path):
    # 建立索引只读取文件头，不预先加载全部体秤波形。
    for encoding in ('utf-8-sig', 'gb18030'):
        metadata = {}
        try:
            with path.open(encoding=encoding) as stream:
                for line in stream:
                    if line.strip().split(',')[0] == 'Index':
                        return metadata
                    if ':' in line or '：' in line:
                        key, value = line.replace('：', ':', 1).split(':', 1)
                        if key.strip() in ('被试ID', '数据单位'):
                            metadata[key.strip()] = value.strip()
        except UnicodeDecodeError:
            continue
        raise ValueError('scale 文件缺少 Index 表头')
    raise ValueError('scale 文件无法解码')


def _scale(path, subject):
    lines = _text(path).splitlines()
    header = next(i for i, line in enumerate(lines)
                  if line.strip().split(',')[0] == 'Index')
    metadata = _scale_header(path)
    reader = csv.reader(lines[header:])
    names = [name.strip() for name in next(reader)]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError('scale 通道表头无效')
    leads = {name: [] for name in names[1:]}
    factor = _unit_factor(metadata['数据单位'])
    for row in reader:
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != len(names):
            raise ValueError('scale 数据列数与表头不一致')
        for name, value in zip(names[1:], row[1:]):
            leads[name].append(_number(value) * factor)
    if not any(leads.values()):
        raise ValueError('scale 文件没有波形数据')
    return _record(path, leads)


def _ecg(path, subject, mode):
    root = ET.parse(path).getroot()
    series = next((element for element in root.findall('.//h:series', _NS)
                   if element.find('h:code', _NS) is not None
                   and element.find('h:code', _NS).get('code') == 'RHYTHM'), None)
    if series is None:
        raise ValueError('XML 缺少主记录 RHYTHM')
    wanted = LIMB_LEADS if mode == 'd6' else ('I',)
    leads = {}
    # 只遍历主记录，避免混入派生的代表心搏。
    for seq in series.findall('h:component/h:sequenceSet/h:component/h:sequence', _NS):
        code_element = seq.find('h:code', _NS)
        if code_element is None:
            continue
        code = code_element.get('code')
        value = seq.find('h:value', _NS)
        name = _LEAD_CODES.get(code)
        if name not in wanted:
            continue
        if name in leads:
            raise ValueError('XML 主记录含重复导联: %s' % name)
        origin = value.find('h:origin', _NS)
        scale = value.find('h:scale', _NS)
        offset = _number(origin.get('value')) * _unit_factor(origin.get('unit'))
        factor = _number(scale.get('value')) * _unit_factor(scale.get('unit'))
        digits = value.find('h:digits', _NS)
        leads[name] = [offset + factor * int(d) for d in digits.text.split()]
    if any(not leads.get(name) for name in wanted):
        raise ValueError('XML 缺少所需导联')
    if len({len(leads[name]) for name in wanted}) != 1:
        raise ValueError('XML 导联采样点数不一致')
    return _record(path, {name: leads[name] for name in wanted})


def _load(parser, path, *args):
    try:
        return parser(path, *args)
    except (ValueError, KeyError, TypeError, AttributeError, StopIteration,
            ET.ParseError) as exc:
        raise DataFormatError('%s: %s' % (path.name, exc)) from exc


class Dataset:
    """比赛数据集读取器；创建实例时只建立文件索引和读取小型用户信息文件。

    data_root 目录下应有 watch/、scale/、Electrocardiograph/。
    userinfo_path 默认是 data_root 目录中的 userinfo.txt。
    波形在调用 read_*、read_subject 或 iter_samples 时才加载。
    """

    def __init__(self, data_root=None, userinfo_path=None):
        self._root = Path(data_root or os.environ.get('COMPETITION_DATA_ROOT')
                          or Path(__file__).parent / 'test_data').resolve()
        self._info_path = Path(userinfo_path or os.environ.get('COMPETITION_USERINFO')
                               or self._root / 'userinfo.txt').resolve()
        if not self._root.is_dir():
            raise FileNotFoundError('数据根目录不存在: %s' % self._root)
        self._userinfo = self._read_userinfo()
        self._files = {key: {} for key in ('watch', 'scale', 'd6', 'd12')}
        for path in sorted((self._root / 'watch').glob('*/*.txt')):
            self._add('watch', path.parent.name, path)
        for path in sorted((self._root / 'scale').glob('*.csv')):
            metadata = _load(_scale_header, path)
            subject = metadata.get('被试ID')
            if not subject:
                raise DataFormatError('%s: 缺少被试ID' % path.name)
            self._add('scale', subject, path)
        for mode in ('d6', 'd12'):
            for path in sorted((self._root / 'Electrocardiograph' / (mode + '_test')).glob('*.xml')):
                self._add(mode, path.stem, path)

    def _add(self, source, subject, path):
        self._files[source].setdefault(subject, []).append(path)

    def _read_userinfo(self):
        result = {}
        reader = csv.DictReader(io.StringIO(_text(self._info_path)))
        required = set(_INFO_FIELDS)
        if not required.issubset(reader.fieldnames or []):
            raise DataFormatError('userinfo 缺少字段: %s' % ', '.join(sorted(required)))
        for row in reader:
            try:
                subject = row['externalid'].strip()
                if not subject or None in row:
                    raise ValueError('被试ID为空或字段数错误')
                # 仅返回约定字段，避免新增内部字段进入参赛者输入。
                info = {key: row[key].strip() for key in _INFO_FIELDS}
                for key in ('height', 'weight', 'age'):
                    info[key] = _number(info[key])
                # 原始 userinfo 含完全相同的重复行，可安全合并；冲突不能覆盖。
                if subject in result and result[subject] != info:
                    raise ValueError('同一被试ID的信息冲突: %s' % subject)
                result[subject] = info
            except (ValueError, TypeError, AttributeError) as exc:
                raise DataFormatError('userinfo 第 %d 行: %s' % (reader.line_num, exc)) from exc
        if not result:
            raise DataFormatError('userinfo 没有被试信息')
        return result

    def list_subjects(self):
        """返回 userinfo 和各设备文件中被试 ID 的并集（排序后）。"""
        return sorted(set(self._userinfo).union(*(set(files) for files in self._files.values())))

    def read_userinfo(self, subject_id=None):
        """不传 ID 返回 {ID: 信息}；指定 ID 返回该被试信息（缺失时抛 KeyError）。"""
        if subject_id is None:
            return {key: dict(value) for key, value in self._userinfo.items()}
        return dict(self._userinfo[subject_id])

    def _read(self, source, subject_id=None):
        if subject_id is not None and subject_id not in self.list_subjects():
            raise KeyError('未知被试ID: %s' % subject_id)
        subjects = self.list_subjects() if subject_id is None else [subject_id]
        result = []
        for subject in subjects:
            for path in self._files[source].get(subject, []):
                if source == 'watch':
                    result.append(_load(_watch, path, subject))
                elif source == 'scale':
                    result.append(_load(_scale, path, subject))
                else:
                    result.append(_load(_ecg, path, subject, source))
        return result

    def read_watch(self, subject_id=None):
        """读取 watch 记录；每条记录的 leads 只有 voltage。"""
        return self._read('watch', subject_id)

    def read_scale(self, subject_id=None):
        """读取体脂秤记录；leads 保留原始通道列名。"""
        return self._read('scale', subject_id)

    def read_ele(self, mode='d6', subject_id=None):
        """mode='d6' 返回六个肢体导联；mode='d12' 只返回 I 导联。"""
        if mode not in ('d6', 'd12'):
            raise ValueError("mode 必须是 'd6' 或 'd12'")
        return self._read(mode, subject_id)

    def read_subject(self, subject_id):
        """返回一位被试的 userinfo、watch、scale、d6 和 d12 输入。

        缺失 userinfo 时 userinfo 为 None；缺失设备时对应记录列表为 []。
        """
        return dict(subject_id=subject_id,
                    userinfo=dict(self._userinfo[subject_id]) if subject_id in self._userinfo else None,
                    watch=self.read_watch(subject_id), scale=self.read_scale(subject_id),
                    ele={mode: self.read_ele(mode, subject_id) for mode in ('d6', 'd12')})

    def iter_samples(self):
        """按编号逐人返回数据，每次仅保留当前被试，推荐用于正式测试。"""
        for subject in self.list_subjects():
            yield self.read_subject(subject)

    def read_all(self):
        """返回 {HNU编号: 全部输入}；一次性加载所有波形，可能占用较多内存。"""
        return {sample['subject_id']: sample for sample in self.iter_samples()}


_default = None


def configure(data_root=None, userinfo_path=None):
    """设置后续模块级 read_* 函数使用的数据目录，并返回 Dataset。

    EXE 中建议传入以可执行文件所在目录为基准计算出的绝对路径，不要写死开发机路径。
    """
    global _default
    dataset = Dataset(data_root, userinfo_path)
    _default = dataset
    return dataset


def _dataset():
    return _default if _default is not None else configure()


def list_subjects():
    return _dataset().list_subjects()


def read_userinfo(subject_id=None):
    return _dataset().read_userinfo(subject_id)


def read_watch(subject_id=None):
    return _dataset().read_watch(subject_id)


def read_scale(subject_id=None):
    return _dataset().read_scale(subject_id)


def read_ele(mode='d6', subject_id=None):
    return _dataset().read_ele(mode, subject_id)


def read_subject(subject_id):
    return _dataset().read_subject(subject_id)


def read_all():
    return _dataset().read_all()


def iter_samples():
    yield from _dataset().iter_samples()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='验证比赛输入文件，输出记录数（不输出波形）')
    parser.add_argument('--data-root')
    parser.add_argument('--userinfo')
    args = parser.parse_args()
    dataset = Dataset(args.data_root, args.userinfo)
    counts = dict(watch=0, scale=0, d6=0, d12=0)
    subjects = 0
    for sample in dataset.iter_samples():
        subjects += 1
        for source in ('watch', 'scale'):
            counts[source] += len(sample[source])
        for source in ('d6', 'd12'):
            counts[source] += len(sample['ele'][source])
    print(json.dumps(dict(subjects=subjects, records=counts), ensure_ascii=True))
