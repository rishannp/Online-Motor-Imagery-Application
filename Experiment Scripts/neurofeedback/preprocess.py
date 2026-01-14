# preprocess.py

# OUR CODE ONLY ACCEPTS 64 Channels as defined in the order below. Do not enter anything else.
import numpy as np
from scipy.signal import butter, filtfilt  # needed for band-pass filtering
from config import METHOD, CSP_CHANNELS

class Preprocessor:
    def __init__(
        self,
        artifact_threshold: float = 3000000.0,
        sampling_rate: float = 256.0,     # NOTE: set to our actual LSL/amp fs
        bp_low_hz: float = 8.0,           # I want 8–12 Hz by default (mu band slice)
        bp_high_hz: float = 30.0,
        bp_order: int = 4,                # 4th-order Butterworth 
        enable_bandpass: bool = True,     # toggle if I want to disable in ablations
        enable_zscore: bool = True        # toggle for z-score normalisation
    ):
        self.artifact_thresh = artifact_threshold
        self.fs = float(sampling_rate)
        self.bp_low_hz = float(bp_low_hz)
        self.bp_high_hz = float(bp_high_hz)
        self.bp_order = int(bp_order)
        self.enable_bandpass = bool(enable_bandpass)
        self.enable_zscore = bool(enable_zscore)

        # Full 64-channel headset layout 
        self.headset_electrodes = [
            'FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
            'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1', 'A2'
        ]

        # 58 shared electrodes used for Stieger training
        self.shared_stieger_electrodes = [
            'FP1', 'FPz', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'
        ]

        # map 64→58 (indices into the raw 64-ch array)
        self.subset_indices = [
            self.headset_electrodes.index(e)
            for e in self.shared_stieger_electrodes
        ]

        # within the 58, pick out FC/C/CP channels for CSP (compatibility)
        self.csp_indices = [
            self.shared_stieger_electrodes.index(ch)
            for ch in CSP_CHANNELS
        ]

        # Precompute band-pass filter coeffs if enabled
        if self.enable_bandpass:
            nyq = 0.5 * self.fs
            if not (0 < self.bp_low_hz < self.bp_high_hz < nyq):
                raise ValueError(
                    f"Band-pass must satisfy 0 < low < high < Nyquist={nyq:.2f} Hz; "
                    f"got low={self.bp_low_hz}, high={self.bp_high_hz}."
                )
            wp = [self.bp_low_hz / nyq, self.bp_high_hz / nyq]
            # Butterworth is maximally flat in passband – good default for EEG band isolation
            self._b_bp, self._a_bp = butter(self.bp_order, wp, btype='bandpass')
        else:
            self._b_bp, self._a_bp = None, None

    def _bandpass(self, window: np.ndarray) -> np.ndarray:
        # zero-phase forward-backward filtering to avoid phase distortions in EEG
        return filtfilt(self._b_bp, self._a_bp, window, axis=0)

    def _zscore(self, window: np.ndarray) -> np.ndarray:
        # per-channel mean/std over samples in this window
        mu = window.mean(axis=0, keepdims=True)
        sd = window.std(axis=0, ddof=1, keepdims=True)
        # I guard against degenerate std (e.g., flat channels) to avoid NaNs
        return (window - mu) / (sd + 1e-8)

    def _artifact_reject(self, window: np.ndarray) -> bool:
        # I keep artifact rejection on raw input (peak-to-peak) to avoid masking transients
        ptp = window.max(axis=0) - window.min(axis=0)
        return np.any(ptp > self.artifact_thresh)

    # === Processing path for AR: keep 64 channels in the same order ===
    def process_keep64(self, window: np.ndarray) -> np.ndarray | None:
        """
        Input window is raw [samples, 64].
        Steps:
          1) artifact rejection on raw window (PTP)
          2) optional band-pass (default 8–12 Hz)
          3) optional per-channel z-score
          4) return [samples, 64] in the original headset order
        """
        if self._artifact_reject(window):
            return None

        w = window
        if self.enable_bandpass:
            w = self._bandpass(w)
        if self.enable_zscore:
            w = self._zscore(w)
        return w  # keep all 64 channels; AR trainer expects original order

    # === Processing path for PLV/CSP: subset to 58 shared electrodes ===
    def process(self, window: np.ndarray) -> np.ndarray | None:
        """
        Input window is raw [samples, 64].
        Steps:
          1) artifact rejection on raw window (PTP)
          2) optional band-pass (default 8–12 Hz)
          3) optional per-channel z-score
          4) subset 64→58 (shared Stieger electrodes)
        """
        if self._artifact_reject(window):
            return None

        w = window
        if self.enable_bandpass:
            w = self._bandpass(w)
        if self.enable_zscore:
            w = self._zscore(w)
        return w[:, self.subset_indices]

# Default preprocessor instance
_pre = Preprocessor()

def preprocess_window(window):
    """
    Input: raw [samples,64].
    Output:
      - [samples,64] when METHOD == 'ar'  (now band-passed + z-scored, original order)
      - [samples,58] when METHOD == 'plv' (processed then subset)
      - [samples,#CSP] when METHOD == 'csp' (processed→subset→CSP subset)
    """
    method = METHOD.lower()

    if method == 'ar':
        # I want AR to use the same clean band-pass + z-score preprocessing
        return _pre.process_keep64(window)

    w = _pre.process(window)
    if w is None:
        return None

    if method == 'csp':
        return w[:, _pre.csp_indices]

    return w
