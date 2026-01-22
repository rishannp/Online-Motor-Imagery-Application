# csp_pipeline.py
import numpy as np
from collections import deque

from config import (
    SAMPLING_RATE,
    CSP_BASELINE_SECONDS,
    CSP_SMOOTH_VOTES,
    CSP_BASELINE_CENTER,
    CSP_ANALOG_OUTPUT,
)
from preprocess import preprocess_window


def _logvar_feats(window_tc: np.ndarray, W: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """
    window_tc: [T, C]
    W: [C, C]
    picks: [k]
    returns: [1, k] log-variance features
    """
    X = window_tc.T.astype(np.float32)  # [C, T]
    Z = (W.T @ X)                       # [C, T]
    Zp = Z[picks, :]                    # [k, T]

    var = np.var(Zp, axis=1).astype(np.float32)
    var /= (np.sum(var) + 1e-12)
    return np.log(var + 1e-12)[None, :]


class CSPPipeline:
    """
    Loads a trained CSP pack (W, picks, clf). Online:

      1) Session baseline (time-accurate):
         - For the first CSP_BASELINE_SECONDS, we DON'T output commands,
           but we DO compute margins to estimate a per-session baseline.

      2) Baseline centering:
         - After baseline, we subtract baseline_mean_margin from each margin
           (optional, CSP_BASELINE_CENTER).

      3) DISCRETE control:
         - We smooth margins by averaging last N values (CSP_SMOOTH_VOTES).
         - Output 0/1 by sign (fixed-speed game control).
    """

    def __init__(self, pack: dict):
        self.W = np.asarray(pack["W"], dtype=np.float32)
        self.picks = np.asarray(pack["picks"], dtype=int)
        self.clf = pack["clf"]  # sklearn estimator (SVM)

        self.baseline_samples = int(float(CSP_BASELINE_SECONDS) * float(SAMPLING_RATE))
        self.seen_samples = 0

        self._baseline_margins = []
        self.baseline_mu = 0.0
        self.baseline_ready = (self.baseline_samples <= 0)

        # store signed (optionally centered) margins
        self.margin_hist = deque(maxlen=max(1, int(CSP_SMOOTH_VOTES)))

    def _margin(self, feats: np.ndarray) -> float:
        """Signed strength: SVM margin. Positive => predicts class 1."""
        if hasattr(self.clf, "decision_function"):
            return float(np.ravel(self.clf.decision_function(feats))[0])

        pred = int(self.clf.predict(feats)[0])
        return 1.0 if pred == 1 else -1.0

    def process(self, window_t64: np.ndarray, n_new: int):
        # IMPORTANT: count only NEW samples streamed in, not the full window length.
        self.seen_samples += int(n_new)

        w_tc = preprocess_window(window_t64)  # [T, Ccsp]
        if w_tc is None:
            return None

        feats = _logvar_feats(w_tc, self.W, self.picks)  # [1, k]
        m_raw = self._margin(feats)

        # ---- Baseline phase: estimate baseline mean margin, but output nothing.
        if not self.baseline_ready and self.seen_samples < self.baseline_samples:
            self._baseline_margins.append(m_raw)
            return None

        # First call after baseline: finalize baseline mean once.
        if not self.baseline_ready:
            self.baseline_mu = float(np.mean(self._baseline_margins)) if self._baseline_margins else 0.0
            self.baseline_ready = True

        # ---- Centering (optional)
        m = (m_raw - self.baseline_mu) if bool(CSP_BASELINE_CENTER) else m_raw

        self.margin_hist.append(m)
        m_avg = float(np.mean(self.margin_hist))

        # ---- DISCRETE output
        # Even if someone accidentally flips CSP_ANALOG_OUTPUT later,
        # we still enforce discrete by default when config is False.
        if bool(CSP_ANALOG_OUTPUT):
            # If you ever re-enable analog intentionally, remove this block.
            pass

        return 1 if m_avg > 0.0 else 0
