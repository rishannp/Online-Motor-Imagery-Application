# preprocess.py
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from config import (
    SAMPLING_RATE,
    APPLY_BANDPASS, BP_LO, BP_HI, BP_ORDER,
    ENABLE_ZSCORE,
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

# ------------------------
# Causal bandpass filtering
# ------------------------
_SOS = None
_ZI_PER_CH = None


def _get_sos(fs: float, lo: float, hi: float, order: int):
    global _SOS
    if _SOS is None:
        nyq = 0.5 * fs
        _SOS = butter(int(order), [float(lo) / nyq, float(hi) / nyq], btype="band", output="sos")
    return _SOS


def reset_preprocess_state():
    """Reset causal filter state (useful between runs/tests).

    Note: for true online operation you typically *do not* reset this between windows.
    """
    global _ZI_PER_CH
    _ZI_PER_CH = None


def _bandpass_ct_causal(x_ct: np.ndarray, fs: float, lo: float, hi: float, order: int):
    """Apply a causal SOS bandpass to x_ct = [C, T], preserving state across calls."""
    global _ZI_PER_CH

    sos = _get_sos(fs, lo, hi, order)
    C, _ = x_ct.shape

    # Lazy init: one zi state per channel
    if _ZI_PER_CH is None or len(_ZI_PER_CH) != C:
        zi0 = sosfilt_zi(sos).astype(np.float32)  # [n_sections, 2]
        _ZI_PER_CH = [zi0.copy() for _ in range(C)]

    y = np.empty_like(x_ct, dtype=np.float32)
    for ch in range(C):
        # Scale zi by first sample to reduce startup transient
        zi = _ZI_PER_CH[ch]
        y_ch, zi_new = sosfilt(sos, x_ct[ch].astype(np.float32), zi=zi * x_ct[ch, 0])
        y[ch] = y_ch
        _ZI_PER_CH[ch] = zi_new

    return y


def bandpass_causal_trial(x_ct: np.ndarray):
    """Causal bandpass over an entire trial (state resets at start of trial).

    This is meant for TRAINING preprocessing so it matches online (causal) behavior.
    Input:  x_ct [C, T]
    Output: y_ct [C, T]
    """
    sos = _get_sos(float(SAMPLING_RATE), float(BP_LO), float(BP_HI), int(BP_ORDER))
    zi0 = sosfilt_zi(sos).astype(np.float32)

    C, _ = x_ct.shape
    y = np.empty_like(x_ct, dtype=np.float32)
    for ch in range(C):
        y_ch, _ = sosfilt(sos, x_ct[ch].astype(np.float32), zi=zi0 * x_ct[ch, 0])
        y[ch] = y_ch
    return y


def _zscore_tc(x_tc: np.ndarray):
    mu = x_tc.mean(axis=0, keepdims=True)
    sd = x_tc.std(axis=0, keepdims=True) + 1e-8
    return ((x_tc - mu) / sd).astype(np.float32)


def preprocess_window_58(window_t64: np.ndarray):
    """Preprocess one window for ONLINE inference (Graph/PLV pipeline).

    window_t64: [T, 64] float
    returns:    [T, 58] float (SHARED_58 order)

    NOTE:
      - Causal, stateful bandpass across calls.
      - If ENABLE_ZSCORE is True, z-scoring is window-local (as before).
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

    return w58_tc


def preprocess_trial_58(trial_t64: np.ndarray):
    """Preprocess a full trial for TRAINING (Graph/PLV pipeline), causal + deterministic.

    This is the training counterpart to preprocess_window_58().

    Key idea:
      - Apply a *causal* bandpass over the *entire trial* with filter state reset at
        the start of the trial (so no cross-trial leakage).
      - Then you segment into windows using WINDOW_SIZE/STEP_SIZE.

    trial_t64: [T, 64]
    returns:   [T, 58]
    """
    if trial_t64 is None or trial_t64.ndim != 2 or trial_t64.shape[1] != 64:
        return None

    x58_ct = trial_t64[:, _IDX_58].T.astype(np.float32)  # [58, T]

    if APPLY_BANDPASS:
        x58_ct = bandpass_causal_trial(x58_ct)

    x58_tc = x58_ct.T
    if ENABLE_ZSCORE:
        x58_tc = _zscore_tc(x58_tc)

    return x58_tc
