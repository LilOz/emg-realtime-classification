import numpy as np
from scipy.stats import kurtosis, skew

def iemg(x):
    """Integrated EMG (IEMG)"""
    return np.sum(np.abs(x))

def var(x):
    """Variance"""
    return np.var(x, ddof=1)

def rms(x):
    """Root Mean Square (RMS)"""
    return np.sqrt(np.mean(x**2))

def ln_rms(x):
    """Log RMS"""
    return np.log(rms(x) + 1e-10)  # Add epsilon to avoid log(0)

def kurt(x):
    """Kurtosis"""
    return kurtosis(x, fisher=True)

def skewness(x):
    """Skewness"""
    return skew(x)

def aac(x):
    """Average Amplitude Change (AAC)"""
    return np.sum(np.abs(np.diff(x))) / len(x)

def mav(x):
    """Mean Absolute Value (MAV)"""
    return np.mean(np.abs(x))

def mav1(x, threshold_ratio=0.25):
    """Modified Mean Absolute Value 1 (MAV1)"""
    th = threshold_ratio * np.max(np.abs(x))
    return np.mean([abs(xi) if abs(xi) > th else 0 for xi in x])

def mav2(x, threshold_ratio=0.25):
    """Modified Mean Absolute Value 2 (MAV2)"""
    th = threshold_ratio * np.max(np.abs(x))
    return np.mean([abs(xi) if abs(xi) < th else 0 for xi in x])

def mavs(x, segment=10):
    """Mean Absolute Value Slope (MAVS)"""
    split = np.array_split(np.abs(x), segment)
    means = [np.mean(seg) for seg in split]
    return np.mean(np.abs(np.diff(means))) if len(means) > 1 else 0

def ssc(x):
    """Slope Sign Changes (SSC)"""
    return np.sum(np.diff(np.sign(np.diff(x))) != 0)

def ssi(x):
    """Simple Square Integral (SSI)"""
    return np.sum(x**2)

def tm(x, order=4):
    """Absolute Temporal Moment (TM)"""
    return np.mean(np.abs(x)**order)

def wamp(x, threshold=10):
    """Willison Amplitude (WAMP)"""
    return np.sum(np.abs(np.diff(x)) >= threshold)

def myop(x, threshold=16):
    """Myopulse Percentage Rate (MYOP)"""
    return np.sum(np.abs(x) > threshold)

def zc(x, threshold=10):
    """Zero Crossings (ZC)"""
    return np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) >= threshold))

feature_extractors = [
    iemg, var, rms, ln_rms, kurt, skewness,
    aac, mav1, mav2, mav, mavs,
    ssc, ssi, tm, wamp, myop, zc
]
