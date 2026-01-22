# graph_pipeline.py
import numpy as np
import time
from collections import deque
from scipy.signal import hilbert

from config import (
    SAMPLING_RATE, STEP_SIZE,
    BASELINE_SECONDS,
    EPSILON,
    SMOOTH_VOTES,
)
from preprocess import preprocess_window_58


def compute_plv_matrix(window_t58: np.ndarray):
    analytic = hilbert(window_t58, axis=0)
    phase = np.angle(analytic)
    u = np.exp(1j * phase)
    T = u.shape[0]
    plv = np.abs((u.conj().T @ u) / max(1, T)).astype(np.float32)
    np.fill_diagonal(plv, 1.0)
    return plv


def plv_transform(plv_raw: np.ndarray, eps: float):
    plv = plv_raw.astype(np.float32).copy()
    np.fill_diagonal(plv, 0.0)
    X = -np.log(1.0 - plv + eps).astype(np.float32)
    np.fill_diagonal(X, 0.0)
    return X


def features_selected(window_t64: np.ndarray, sel_e: np.ndarray, sel_n: np.ndarray, eps: float):
    w58 = preprocess_window_58(window_t64)
    if w58 is None:
        return None

    plv = compute_plv_matrix(w58)
    X = plv_transform(plv, eps=eps)

    triu = np.triu_indices(X.shape[0], k=1)
    edges = X[triu].astype(np.float32)                    # [E]
    nodes = np.sum(np.abs(X), axis=1).astype(np.float32)  # [58]

    Fe = edges[sel_e] if sel_e.size > 0 else np.zeros((0,), dtype=np.float32)
    Fn = nodes[sel_n] if sel_n.size > 0 else np.zeros((0,), dtype=np.float32)

    f = np.concatenate([Fe, Fn], axis=0).astype(np.float32)
    return f[None, :]  # [1, d]


class GraphMLPipeline:
    def __init__(self, pack: dict):
        self.clf = pack["clf"]  # sklearn Pipeline or estimator
        self.sel_e = np.asarray(pack["sel_edge_idx"], dtype=int)
        self.sel_n = np.asarray(pack["sel_node_idx"], dtype=int)

        self.baseline_samples = int(float(BASELINE_SECONDS) * float(SAMPLING_RATE))
        self.warmup_samples = int(float(getattr(__import__("config"), "BASELINE_WARMUP_SECONDS", 0.0)) * float(SAMPLING_RATE))
        self.seen_samples = 0

        # Margin + baseline centering
        cfg = __import__("config")
        self.use_margin = bool(getattr(cfg, "USE_MARGIN_OUTPUT", True))
        self.use_center = bool(getattr(cfg, "ENABLE_BASELINE_CENTERING", True))

        self.baseline_done = False
        self._baseline_margins = []
        self._baseline_bias = 0.0

        self.margin_hist = deque(maxlen=max(1, int(SMOOTH_VOTES)))

        self._dbg_last_print = 0.0
        self._dbg_print_interval = 1.0  # seconds


    def _score_margin(self, X: np.ndarray) -> float:
        """Return a signed scalar score. Positive => class 1, negative => class 0."""
        # Prefer decision_function (SVM, linear models, etc.)
        if hasattr(self.clf, "decision_function"):
            m = self.clf.decision_function(X)
            m = np.asarray(m).ravel()
            return float(m[0])

        # Fall back to predict_proba if available (LogReg, some SVCs)
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(X)
            p = np.asarray(p).reshape(-1, p.shape[-1])
            if p.shape[1] >= 2:
                p1 = float(p[0, 1])
            else:
                p1 = float(p[0, 0])
            # Convert to logit so the sign is meaningful and scale is less squashed than p-0.5.
            p1 = min(max(p1, 1e-6), 1.0 - 1e-6)
            return float(np.log(p1 / (1.0 - p1)))

        # Last resort: hard label as +/-1
        pred = int(self.clf.predict(X)[0])
        return 1.0 if pred == 1 else -1.0

    def process(self, window_t64: np.ndarray, n_new: int = None):
        """Run inference on the most recent WINDOW_SIZE samples.
    
        window_t64: [T, 64]
        n_new:      how many *new* samples have advanced since the previous inference.
                    If None, defaults to STEP_SIZE.
        """
        self.seen_samples += int(STEP_SIZE if n_new is None else n_new)
    
        # Still in baseline *time window* (we keep returning None so the game stays centered)
        if self.seen_samples < self.baseline_samples:
            # Optionally accumulate baseline margins for centering (after warmup)
            if self.use_center and self.use_margin and (self.seen_samples >= self.warmup_samples):
                Xb = features_selected(window_t64, self.sel_e, self.sel_n, eps=float(EPSILON))
                if Xb is not None:
                    self._baseline_margins.append(self._score_margin(Xb))
    
            # -------- DEBUG PRINT (once per second) --------
            now = time.time()
            if now - self._dbg_last_print >= self._dbg_print_interval:
                remaining = max(0.0, (self.baseline_samples - self.seen_samples) / float(SAMPLING_RATE))
                if len(self._baseline_margins) > 0:
                    mean_m = float(np.mean(self._baseline_margins))
                else:
                    mean_m = 0.0
    
                # show whether we're still warming up (i.e., not yet collecting margins)
                collecting = (
                    self.use_center and self.use_margin and (self.seen_samples >= self.warmup_samples)
                )
                tag = "collecting" if collecting else "warmup"
    
                print(f"[BASELINE] remaining={remaining:.1f}s | mean_margin={mean_m:+.3f} | {tag}")
                self._dbg_last_print = now
            # ----------------------------------------------
    
            return None
    
        # We just crossed baseline: lock bias once
        if (not self.baseline_done) and self.use_center and self.use_margin:
            if len(self._baseline_margins) > 0:
                self._baseline_bias = float(np.mean(self._baseline_margins))
            else:
                self._baseline_bias = 0.0
    
            self.baseline_done = True
            self.margin_hist.clear()
    
            # -------- DEBUG PRINT (baseline finished) -----
            print(f"[BASELINE DONE] bias={self._baseline_bias:+.3f} (centering enabled)")
            # ----------------------------------------------
    
            # First post-baseline call returns None to avoid an abrupt jump.
            return None
    
        X = features_selected(window_t64, self.sel_e, self.sel_n, eps=float(EPSILON))
        if X is None:
            return None
    
        if self.use_margin:
            m = self._score_margin(X)
            if self.use_center:
                m = float(m - self._baseline_bias)
    
            self.margin_hist.append(float(m))
    
            # Smooth in margin-space, then take the sign => discrete command, fixed speed.
            mbar = float(np.mean(self.margin_hist))
            return 1 if mbar > 0.0 else 0
    
        # Fallback: original hard-vote mode
        pred = int(self.clf.predict(X)[0])  # 0/1
        self.margin_hist.append(1.0 if pred == 1 else -1.0)
        mbar = float(np.mean(self.margin_hist))
        return 1 if mbar > 0.0 else 0
