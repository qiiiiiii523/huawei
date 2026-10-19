"""赛题 1 可选预处理参考，不属于必须调用的输入或输出接口。

处理顺序：100 Hz 四阶零相位低通 -> 估计并减去 0.3 Hz 基线 -> 抗混叠降采样
到 500 Hz。目的是减弱高频噪声和基线漂移，同时尽量保留心电波形特征。

输入 raw 为形状 (12, N) 的数值数组，导联顺序固定为 LEADS；fs 是输入采样率，
且不得低于 500 Hz。函数处理全部导联（包括 I），返回新的 (12, M) 数组，不修改
原数组，也不自动截取或补齐为两分钟。正式提交前仍需由参赛程序选择正确的
60000 点，并通过 output.py 保存。

示例：

    clean = preprocess_signal(raw_12_leads, fs=1000)

本参考依赖 numpy 和 scipy；参赛者可以采用其他合理方法，也可以不使用本文件。
"""
import math

LEADS = ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')
FS = 500

def preprocess_signal(raw, fs):
    """对全部 12 导联做低通、基线校正并降采样到 500 Hz。

    raw 为 (12, N)，导联顺序见 LEADS，至少 20 秒；输入 fs 不低于 500 Hz。
    返回新数组，不修改输入，不截取/补齐两分钟。依赖 numpy 和 scipy。
    """
    from fractions import Fraction
    import numpy as np
    from scipy import signal

    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(fs) or fs < FS:
        raise ValueError('输入采样率必须是有限数值且不低于 500 Hz')
    if x.ndim != 2 or x.shape[0] != len(LEADS) or x.shape[1] < round(fs * 20):
        raise ValueError('需要 (12, N) 数组，至少 20 秒')
    if not np.isfinite(x).all():
        raise ValueError('信号须为有限数值')
    pad = min(round(fs * 10), x.shape[1] - 2)
    # 零相位滤波避免普通单向滤波造成波形时间偏移。
    filtered = signal.sosfiltfilt(
        signal.butter(4, 100., 'lowpass', fs=fs, output='sos'), x, axis=-1, padlen=pad)
    # 0.3 Hz 低通结果作为缓慢变化的基线，减去后抑制上下漂移。
    baseline = signal.sosfiltfilt(
        signal.butter(2, .3, 'lowpass', fs=fs, output='sos'), filtered, axis=-1, padlen=pad)
    ratio = Fraction(FS / fs).limit_denominator(10000)
    if not math.isclose(fs * ratio.numerator / ratio.denominator, FS, rel_tol=1e-12):
        raise ValueError('无法精确转换到 500 Hz')
    # resample_poly 在降采样时包含抗混叠处理。
    result = signal.resample_poly(filtered - baseline, ratio.numerator, ratio.denominator,
                                  axis=-1, window=('kaiser', 5.0), padtype='line')
    if result.shape != (len(LEADS), math.ceil(x.shape[1] * ratio.numerator / ratio.denominator)):
        raise ValueError('重采样后的长度错误')
    if not np.isfinite(result).all():
        raise ValueError('预处理结果含非有限值')
    return result
