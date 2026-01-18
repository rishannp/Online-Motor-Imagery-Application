# preprocess.py
import numpy as np
from scipy.signal import butter, filtfilt

from config import (
    SAMPLING_RATE,
    APPLY_BANDPASS, BP_LO, BP_HI, BP_ORDER,
    ENABLE_ZSCORE,
    CSP_CHANNELS,
)

# Channel lists must match what you record in session_data.pkl / LSL order
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

def idx_map(src_names, keep_names):
    src_idx = {ch: i for i, ch in enumerate(src_names)}
    missing = [ch for ch in keep_names if ch not in src_idx]
    if missing:
        raise RuntimeError(f"Missing channels in src list: {missing}")
    return [src_idx[ch] for ch in keep_names]

_IDX_58 = np.array(idx_map(HEADSET_64, SHARED_58), dtype=int)
_IDX_CSP = np.array(idx_map(SHARED_58, CSP_CHANNELS), dtype=int)

def _bandpass_ct(x_ct: np.ndarray, fs: float, lo: float, hi: float, order: int):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, x_ct, axis=1)

def _zscore_tc(x_tc: np.ndarray):
    mu = x_tc.mean(axis=0, keepdims=True)
    sd = x_tc.std(axis=0, keepdims=True) + 1e-8
    return ((x_tc - mu) / sd).astype(np.float32)

def preprocess_window(window_t64: np.ndarray):
    """
    window_t64: [T, 64] float
    returns:    [T, Ccsp] float (CSP channels in CSP_CHANNELS order)
    """
    if window_t64 is None or window_t64.ndim != 2 or window_t64.shape[1] != 64:
        return None

    w58_ct = window_t64[:, _IDX_58].T.astype(np.float32)  # [58, T]

    if APPLY_BANDPASS:
        w58_ct = _bandpass_ct(w58_ct, float(SAMPLING_RATE), float(BP_LO), float(BP_HI), int(BP_ORDER))

    w58_tc = w58_ct.T  # [T, 58]

    if ENABLE_ZSCORE:
        w58_tc = _zscore_tc(w58_tc)

    w_csp_tc = w58_tc[:, _IDX_CSP]  # [T, Ccsp]
    return w_csp_tc
