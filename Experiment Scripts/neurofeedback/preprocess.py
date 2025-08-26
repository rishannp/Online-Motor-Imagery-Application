# preprocess.py

# OUR CODE ONLY ACCEPTS 64 Channels as defined in the order below. Do not enter anything else. 
import numpy as np
from config import METHOD, CSP_CHANNELS

class Preprocessor:
    def __init__(self, artifact_threshold=3000.0):
        self.artifact_thresh = artifact_threshold

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

        # map 64→58
        self.subset_indices = [
            self.headset_electrodes.index(e)
            for e in self.shared_stieger_electrodes
        ]

        # within the 58, pick out FC/C/CP channels for CSP (compatibility)
        self.csp_indices = [
            self.shared_stieger_electrodes.index(ch)
            for ch in CSP_CHANNELS
        ]

    def process(self, window: np.ndarray) -> np.ndarray:
        # Simple artifact rejection
        ptp = window.max(axis=0) - window.min(axis=0)
        if np.any(ptp > self.artifact_thresh):
            return None

        # Trainer runs on raw 64-ch; PLV/CSP paths would subset
        return window[:, self.subset_indices]  # available if you switch modes

_pre = Preprocessor()

def preprocess_window(window):
    """
    Input: raw [samples,64].
    Output:
      - [samples,64] raw if METHOD == 'ar'  (we'll index channels directly)
      - [samples,58] when METHOD == 'plv'
      - [samples,#CSP] when METHOD == 'csp'
    """
    if METHOD.lower() == 'ar':
        return window  # neurofeedback trainer wants raw channel order

    w = _pre.process(window)
    if w is None:
        return None
    if METHOD.lower() == 'csp':
        return w[:, _pre.csp_indices]
    return w
