import numpy as np
from scipy.stats import kurtosis, skew
from statsmodels.tsa.ar_model import AutoReg

def iemg(x):
    """Integrated EMG (IEMG)"""
    return np.sum(np.abs(x))

def msv(x):
    """Mean Squared Value (MSV)"""
    return np.mean(x**2)

def var(x):
    """Variance"""
    return np.var(x, ddof=1)

def rms(x):
    """Root Mean Square (RMS)"""
    return np.sqrt(np.mean(x**2))

def ln_rms(x):
    """Log RMS"""
    return np.log(rms(x))

def kurt(x):
    """Kurtosis"""
    return kurtosis(x, fisher=True)  # Subtracts 3 by default

def skewness(x):
    """Skewness"""
    return skew(x)

def arm_coefficients(x, order=6):
    """Auto-Regressive Model (ARM) coefficients"""
    model = AutoReg(x, lags=order, old_names=False)
    model_fit = model.fit()
    return model_fit.params  # Includes intercept if fit_intercept=True
