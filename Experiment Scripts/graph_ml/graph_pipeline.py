# graph_pipeline.py
import numpy as np
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
        self.clf = pack["clf"]  # sklearn Pipeline
        self.sel_e = np.asarray(pack["sel_edge_idx"], dtype=int)
        self.sel_n = np.asarray(pack["sel_node_idx"], dtype=int)

        self.baseline_samples = int(float(BASELINE_SECONDS) * float(SAMPLING_RATE))
        self.seen_samples = 0

        self.vote_hist = deque(maxlen=max(1, int(SMOOTH_VOTES)))

    def process(self, window_t64: np.ndarray, n_new: int = None):
        """Run inference on the most recent WINDOW_SIZE samples.

        window_t64: [T, 64]
        n_new:      how many *new* samples have advanced since the previous inference.
                    If None, defaults to STEP_SIZE.
        """
        self.seen_samples += int(STEP_SIZE if n_new is None else n_new)
        if self.seen_samples < self.baseline_samples:
            return None

        X = features_selected(window_t64, self.sel_e, self.sel_n, eps=float(EPSILON))
        if X is None:
            return None

        pred = int(self.clf.predict(X)[0])  # 0/1
        self.vote_hist.append(pred)

        if len(self.vote_hist) == 1:
            return pred

        return 1 if sum(self.vote_hist) > (len(self.vote_hist) / 2.0) else 0
