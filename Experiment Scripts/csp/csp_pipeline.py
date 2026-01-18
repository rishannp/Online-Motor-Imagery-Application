# csp_pipeline.py
import numpy as np
from collections import deque

from config import (
    SAMPLING_RATE, STEP_SIZE,
    CSP_BASELINE_SECONDS,
    CSP_SMOOTH_VOTES,
)
from preprocess import preprocess_window

def _logvar_feats(window_tc: np.ndarray, W: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """
    window_tc: [T, C]
    W: [C, C]
    picks: [k]
    """
    X = window_tc.T.astype(np.float32)   # [C, T]
    Z = (W.T @ X)                        # [C, T]
    Zp = Z[picks, :]                     # [k, T]
    var = np.var(Zp, axis=1).astype(np.float32)
    var /= (np.sum(var) + 1e-12)
    return np.log(var + 1e-12)[None, :]

class CSPPipeline:
    """
    process(window_t64):
      returns None during baseline, else 0/1
    """
    def __init__(self, pack: dict):
        self.W = np.asarray(pack["W"], dtype=np.float32)
        self.picks = np.asarray(pack["picks"], dtype=int)
        self.clf = pack["clf"]  # sklearn Pipeline

        self.baseline_samples = int(float(CSP_BASELINE_SECONDS) * float(SAMPLING_RATE))
        self.seen_samples = 0

        self.vote_hist = deque(maxlen=max(1, int(CSP_SMOOTH_VOTES)))

    def process(self, window_t64: np.ndarray):
        # baseline gating (match game behavior exactly)
        self.seen_samples += int(STEP_SIZE)
        if self.seen_samples < self.baseline_samples:
            return None

        w_tc = preprocess_window(window_t64)  # [T, Ccsp]
        if w_tc is None:
            return None

        feats = _logvar_feats(w_tc, self.W, self.picks)  # [1, k]
        pred = int(self.clf.predict(feats)[0])           # 0/1

        self.vote_hist.append(pred)
        if len(self.vote_hist) == 1:
            return pred

        return 1 if sum(self.vote_hist) > (len(self.vote_hist) / 2.0) else 0
