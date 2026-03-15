# preprocess.py
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import threading

from config import (
    SAMPLING_RATE,
    APPLY_BANDPASS, BP_LO, BP_HI, BP_ORDER,
    ENABLE_ZSCORE,
    CSP_CHANNELS,
)

# Channel lists must match what you record in session_data.pkl / LSL order.
HEADSET_64 = (
    "FP1 FPz FP2 AF7 AF3 AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 "
    "FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 T7 C5 C3 C1 Cz C2 C4 C6 "
    "T8 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 "
    "PO7 PO3 POz PO4 PO8 O1 Oz O2 F9 F10 A1 A2"
).split()

SHARED_58 = (
    "FP1 FPz FP2 AF3 AF4 F7 F5 F3 F1 Fz F2 F4 F6 F8 "
    "FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 T7 C5 C3 C1 Cz C2 C4 C6 "
    "T8 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 "
    "PO7 PO3 POz PO4 PO8 O1 Oz O2"
).split()

def idx_map(src_names, keep_names):
    src_idx = {ch: i for i, ch in enumerate(src_names)}
    missing = [ch for ch in keep_names if ch not in src_idx]
    if missing:
        raise RuntimeError(f"Missing channels in src list: {missing}")
    return [src_idx[ch] for ch in keep_names]

_IDX_58 = np.array(idx_map(HEADSET_64, SHARED_58), dtype=int)
_IDX_CSP = np.array(idx_map(SHARED_58, CSP_CHANNELS), dtype=int)

# ------------------------
# Causal bandpass filtering
# ------------------------
_SOS = None
_ZI_PER_CH = None
_BP_LOCK = threading.Lock()

def _get_sos(fs: float, lo: float, hi: float, order: int):
    global _SOS
    if _SOS is None:
        nyq = 0.5 * fs
        _SOS = butter(int(order), [float(lo) / nyq, float(hi) / nyq], btype="band", output="sos")
    return _SOS

def reset_preprocess_state():
    """Reset online causal filter state.

    For continuous online operation, do NOT call this at trial boundaries.
    Safe to call between runs or before starting inference.
    """
    global _ZI_PER_CH
    with _BP_LOCK:
        _ZI_PER_CH = None

def _bandpass_ct_causal(x_ct: np.ndarray, fs: float, lo: float, hi: float, order: int):
    """Apply a causal SOS bandpass to x_ct = [C, T], preserving state across calls."""
    global _ZI_PER_CH

    sos = _get_sos(fs, lo, hi, order)
    C, _ = x_ct.shape

    with _BP_LOCK:
        if _ZI_PER_CH is None or len(_ZI_PER_CH) != C:
            zi0 = sosfilt_zi(sos).astype(np.float32)  # [n_sections, 2]
            _ZI_PER_CH = [zi0.copy() for _ in range(C)]

        y = np.empty_like(x_ct, dtype=np.float32)
        for ch in range(C):
            zi = _ZI_PER_CH[ch]
            y_ch, zi_new = sosfilt(sos, x_ct[ch].astype(np.float32), zi=zi * x_ct[ch, 0])
            y[ch] = y_ch
            _ZI_PER_CH[ch] = zi_new

    return y

def bandpass_causal_trials_continuous(trials_ct, fs=None, lo=None, hi=None, order=None):
    """Training helper: causal bandpass across a list of trials with continuous state.

    trials_ct: list of [C, T] arrays (same C)
    returns:   list of [C, T] arrays, filtered sequentially sharing filter memory

    This does NOT touch the online global state (_ZI_PER_CH).
    """
    if not trials_ct:
        return []

    fs = float(SAMPLING_RATE if fs is None else fs)
    lo = float(BP_LO if lo is None else lo)
    hi = float(BP_HI if hi is None else hi)
    order = int(BP_ORDER if order is None else order)

    sos = _get_sos(fs, lo, hi, order)
    C = int(trials_ct[0].shape[0])

    zi0 = sosfilt_zi(sos).astype(np.float32)
    zi_per_ch = [zi0.copy() for _ in range(C)]

    out = []
    for k, x_ct in enumerate(trials_ct):
        if x_ct is None or x_ct.ndim != 2 or x_ct.shape[0] != C:
            raise RuntimeError(f"Trial {k} has invalid shape: {None if x_ct is None else x_ct.shape}")

        y = np.empty_like(x_ct, dtype=np.float32)
        for ch in range(C):
            x = x_ct[ch].astype(np.float32)
            zi = zi_per_ch[ch]

            # Only scale zi at the very beginning of the entire continuous stream
            if k == 0:
                y_ch, zi_new = sosfilt(sos, x, zi=zi * x[0])
            else:
                y_ch, zi_new = sosfilt(sos, x, zi=zi)

            y[ch] = y_ch
            zi_per_ch[ch] = zi_new

        out.append(y)

    return out

def _zscore_tc(x_tc: np.ndarray):
    mu = x_tc.mean(axis=0, keepdims=True)
    sd = x_tc.std(axis=0, keepdims=True) + 1e-8
    return ((x_tc - mu) / sd).astype(np.float32)

def preprocess_window(window_t64: np.ndarray):
    """Preprocess one window for ONLINE inference.

    window_t64: [T, 64] float
    returns:    [T, Ccsp] float (CSP channels in CSP_CHANNELS order)
    """
    if window_t64 is None or window_t64.ndim != 2 or window_t64.shape[1] != 64:
        return None

    w58_ct = window_t64[:, _IDX_58].T.astype(np.float32)  # [58, T]

    if APPLY_BANDPASS:
        w58_ct = _bandpass_ct_causal(
            w58_ct,
            float(SAMPLING_RATE),
            float(BP_LO),
            float(BP_HI),
            int(BP_ORDER),
        )

    w58_tc = w58_ct.T  # [T, 58]

    if ENABLE_ZSCORE:
        w58_tc = _zscore_tc(w58_tc)

    w_csp_tc = w58_tc[:, _IDX_CSP]  # [T, Ccsp]
    return w_csp_tc
