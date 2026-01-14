import numpy as np
from scipy.signal import butter, filtfilt

from config import (
    SAMPLING_RATE, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER,
    ARTIFACT_PTP_UV, ENABLE_ZSCORE
)

HEADSET_64 = [
    'FP1','FPz','FP2','AF7','AF3','AF4','AF8','F7','F5','F3',
    'F1','Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz',
    'FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7',
    'PO3','POz','PO4','PO8','O1','Oz','O2','F9','F10','A1','A2'
]

SHARED_58 = [
    'FP1','FPz','FP2','AF3','AF4','F7','F5','F3','F1','Fz',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7',
    'PO3','POz','PO4','PO8','O1','Oz','O2'
]

_SRC_IDX = {ch: i for i, ch in enumerate(HEADSET_64)}
SUBSET_IDX = [ _SRC_IDX[ch] for ch in SHARED_58 ]

def _bandpass(x: np.ndarray, fs: float):
    nyq = 0.5 * fs
    lo, hi = BP_LOW_HZ / nyq, BP_HIGH_HZ / nyq
    b, a = butter(BP_ORDER, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=0)

def _zscore(x: np.ndarray):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    return (x - mu) / (sd + 1e-8)

def preprocess_window(window_64: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray | None:
    """
    Input:  window_64 [T, 64]
    Output: window_58 [T, 58] after:
      - artifact reject (ptp)
      - bandpass 8–30
      - optional z-score
      - subset to shared 58
    """
    if window_64.ndim != 2 or window_64.shape[1] != 64:
        return None

    ptp = window_64.max(axis=0) - window_64.min(axis=0)
    if np.any(ptp > ARTIFACT_PTP_UV):
        return None

    w = _bandpass(window_64, fs)
    if ENABLE_ZSCORE:
        w = _zscore(w)

    return w[:, SUBSET_IDX]
