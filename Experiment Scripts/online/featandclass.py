# featandclass.py


import os
import re
import json
from glob import glob
from collections import deque

import numpy as np
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

from config import (
    METHOD, ADAPTATION, ADAPT_N,
    CSP_LDA_PKL, GAT_MODEL_PT,   # <-- resolved by config.py
    SUBJECT_DIR, SESSION_DIR
)
from preprocess import Preprocessor  # for channel indexing


# ──────────────────────────────────────────────────────────────────────────────
# Utility: PLV + graph building
# ──────────────────────────────────────────────────────────────────────────────

def plvfcn(eegData: np.ndarray) -> np.ndarray:
    """
    Compute PLV adjacency (C x C) from window (T x C).
    """
    phase = np.angle(sig.hilbert(eegData, axis=0))
    C = eegData.shape[1]
    plv = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        for j in range(i + 1, C):
            d = phase[:, j] - phase[:, i]
            v = np.abs(np.exp(1j * d).mean())
            plv[i, j] = plv[j, i] = v
    np.fill_diagonal(plv, 1.0)
    return plv


def threshold_graph_edges(plv: np.ndarray, topk_percent: float = 0.4) -> torch.Tensor:
    """
    Keep top-k% strongest undirected edges (exclude diagonal) and add self-loops.
    Returns edge_index (2, E) as torch.long
    """
    A = plv.copy()
    np.fill_diagonal(A, 0.0)
    triu = np.triu_indices(A.shape[0], k=1)
    weights = A[triu]
    k = max(1, int(len(weights) * topk_percent))
    idx = np.argpartition(weights, -k)[-k:]
    row, col = triu[0][idx], triu[1][idx]
    ei = np.hstack([np.stack([row, col]), np.stack([col, row])])  # (2, 2k)
    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=A.shape[0])
    return edge_index


# ──────────────────────────────────────────────────────────────────────────────
# Utility: find latest ONLINE finetuned models for this subject
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_finetuned_model_path(model_type: str) -> str | None:
    """
    model_type: 'gat' or 'csp_lda'
    Searches the subject's ONLINE results tree:
        SUBJECT_DIR/Session_*/gat_finetuned_*.pt
        SUBJECT_DIR/Session_*/csp_lda_finetuned_*.pkl
    Returns newest by mtime, or None.
    """
    if model_type == "gat":
        pattern = os.path.join(SUBJECT_DIR, "Session_*", "gat_finetuned_*.pt")
    elif model_type == "csp_lda":
        pattern = os.path.join(SUBJECT_DIR, "Session_*", "csp_lda_finetuned_*.pkl")
    else:
        return None

    files = glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def log_model_path(filename: str) -> None:
    log_file = os.path.join(SESSION_DIR, "model_log.json")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log = json.load(f)
    else:
        log = []
    log.append(filename)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# GAT model
# ──────────────────────────────────────────────────────────────────────────────

class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, h1, heads=heads, concat=True, dropout=dropout)
        self.gn1   = GraphNorm(h1 * heads)
        self.conv2 = GATv2Conv(h1 * heads, h2, heads=heads, concat=True, dropout=dropout)
        self.gn2   = GraphNorm(h2 * heads)
        self.conv3 = GATv2Conv(h2 * heads, h3, heads=heads, concat=False, dropout=dropout)
        self.gn3   = GraphNorm(h3)
        self.lin   = nn.Linear(h3, 2)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, "batch", None)
        if batch is None:
            # Single graph; create a dummy batch of zeros
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class BCIPipeline:
    """
    Online MI pipeline.
    - METHOD 'csp': uses CSP filters + LDA classifier.
    - METHOD 'plv': builds PLV graphs and runs a GAT.
    Initialization prefers the latest ONLINE finetuned model (if ADAPTATION=True).
    Otherwise it loads the OFFLINE-trained files resolved in config.py.
    """

    def __init__(self, method=METHOD, fs=256):
        self.method   = method.lower()
        self.fs       = fs
        self.adaptive = ADAPTATION
        self.adapt_N  = ADAPT_N

        self._win_buf: list[np.ndarray] = []
        self._lab_buf: list[int] = []
        self.latest_plv: np.ndarray | None = None

        if self.method == 'csp':
            self._init_csp_lda()
        elif self.method == 'plv':
            self._init_gat()
        else:
            raise ValueError(f"Unknown method {self.method!r}")

    # ── init helpers ─────────────────────────────────────────────────────

    def _init_csp_lda(self):
        """
        Load CSP filters + LDA params.
        Priority:
            1) latest ONLINE finetuned (if adaptive)
            2) offline-trained pack from CSP_LDA_PKL (resolved by config.py)
        """
        saved = None
        if self.adaptive:
            latest = get_latest_finetuned_model_path('csp_lda')
            if latest and os.path.isfile(latest):
                with open(latest, 'rb') as f:
                    saved = pickle_safe_load(f=None, fileobj=f)
        if saved is None:
            with open(CSP_LDA_PKL, 'rb') as f:
                saved = pickle_safe_load(f=None, fileobj=f)

        filters   = saved['filters']
        coef      = saved['lda_coef']
        intercept = saved['lda_intercept']

        # Reconstruct CSP & LDA
        self.csp = CSP(n_components=len(filters), reg=None, log=True, norm_trace=False)
        # mne's CSP expects fitted attributes; setting filters_ is enough for transform
        self.csp.filters_ = np.asarray(filters)

        self.lda = LinearDiscriminantAnalysis()
        self.lda.coef_      = np.asarray(coef)
        self.lda.intercept_ = np.asarray(intercept)

    def _init_gat(self):
        """
        Load GATv2 from state dict.
        Priority:
            1) latest ONLINE finetuned (if adaptive)
            2) offline-trained state dict from GAT_MODEL_PT (resolved by config.py)
        """
        model_path = None
        if self.adaptive:
            latest = get_latest_finetuned_model_path('gat')
            if latest and os.path.isfile(latest):
                model_path = latest
        if model_path is None:
            model_path = GAT_MODEL_PT

        sd = torch.load(model_path, map_location='cpu')

        # infer dims from state dict
        heads = sd['conv1.att'].shape[1]
        h1 = sd['conv1.att'].shape[2]
        h2 = sd['conv2.att'].shape[2]
        h3 = sd['conv3.att'].shape[2]
        in_ch = sd['conv1.lin_l.weight'].shape[1]

        self.gat = SimpleGAT(in_ch, h1, h2, h3, heads)
        self.gat.load_state_dict(sd)
        self.gat.eval()

    # ── inference ────────────────────────────────────────────────────────

    def predict(self, window: np.ndarray) -> int:
        """
        window: (T, C) already preprocessed to 58 channels (PLV) or CSP channels.
        Returns int class {0,1}
        """
        if self.method == 'csp':
            # MNE CSP expects (n_epochs, n_channels, n_times)
            arr = window.T[np.newaxis, ...]              # (1, C, T)
            feat = self.csp.transform(arr)               # (1, n_comp)
            # LDA expects 2D array (n_samples, n_features)
            pred = int(self.lda.predict(feat)[0])
            return pred

        # PLV + GAT
        plv = plvfcn(window)                              # (C, C) in [0,1]
        self.latest_plv = plv.copy()

        # Edge weights transform (match training): -log(1-PLV+eps)
        adj = -np.log(1.0 - plv + 1e-6)
        edge_index = threshold_graph_edges(adj, topk_percent=0.4)
        x = torch.eye(adj.shape[0], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            out = self.gat(data)                          # (1, 2)
        return int(out.argmax(dim=1).item())

    # ── online adaptation ────────────────────────────────────────────────

    def adapt(self):
        """
        If enough labeled windows are buffered (>= adapt_N), perform a small finetune
        and save into SESSION_DIR. Clears buffers after saving.
        """
        if not self.adaptive or len(self._win_buf) < self.adapt_N:
            return

        if self.method == 'csp':
            self._adapt_csp_lda()
        else:
            self._adapt_gat()

        self._win_buf.clear()
        self._lab_buf.clear()

    def _adapt_csp_lda(self):
        # Re-fit CSP & LDA on buffered windows
        X = np.stack(self._win_buf)            # (N, T, C)
        y = np.array(self._lab_buf)            # (N,)

        # MNE CSP fit expects (n_epochs, n_channels, n_times)
        self.csp.fit(np.transpose(X, (0, 2, 1)), y)
        feats = self.csp.transform(np.transpose(X, (0, 2, 1)))  # (N, n_comp)
        self.lda.fit(feats, y)

        # Save new CSP+LDA pack
        os.makedirs(SESSION_DIR, exist_ok=True)
        existing = glob(os.path.join(SESSION_DIR, "csp_lda_finetuned_*.pkl"))
        nums = [int(m.group(1)) for f in existing if (m := re.search(r'finetuned_(\d+)', f))]
        next_num = max(nums) + 1 if nums else 1
        save_path = os.path.join(SESSION_DIR, f"csp_lda_finetuned_{next_num}.pkl")

        pack = {
            'filters': self.csp.filters_,
            'lda_coef': self.lda.coef_,
            'lda_intercept': self.lda.intercept_,
        }
        with open(save_path, 'wb') as f:
            import pickle
            pickle.dump(pack, f)
        log_model_path(save_path)

    def _adapt_gat(self):
        # Build PLV-graph dataset from buffered windows
        datas = []
        for w, lbl in zip(self._win_buf, self._lab_buf):
            plv = plvfcn(w)
            adj = -np.log(1.0 - plv + 1e-6)
            ei  = threshold_graph_edges(adj, topk_percent=0.4)
            x   = torch.eye(adj.shape[0], dtype=torch.float32)
            datas.append(Data(x=x, edge_index=ei, y=torch.tensor([lbl], dtype=torch.long)))

        loader = DataLoader(datas, batch_size=16, shuffle=True)

        opt = torch.optim.Adam(self.gat.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss()

        self.gat.train()
        for _ in range(5):  # a quick few epochs
            for batch in loader:
                opt.zero_grad()
                out  = self.gat(batch)
                loss = crit(out, batch.y)
                loss.backward()
                opt.step()
        self.gat.eval()

        # Save new state dict
        os.makedirs(SESSION_DIR, exist_ok=True)
        existing = glob(os.path.join(SESSION_DIR, "gat_finetuned_*.pt"))
        nums = [int(m.group(1)) for f in existing if (m := re.search(r'finetuned_(\d+)', f))]
        next_num = max(nums) + 1 if nums else 1
        save_path = os.path.join(SESSION_DIR, f"gat_finetuned_{next_num}.pt")
        torch.save(self.gat.state_dict(), save_path)
        log_model_path(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers used elsewhere
# ──────────────────────────────────────────────────────────────────────────────

def pickle_safe_load(path: str | None = None, fileobj=None):
    """
    Tiny helper to load pickle either from path or file-like object with
    restricted loader (safety is limited since we trust our own files).
    """
    import pickle
    if fileobj is not None:
        return pickle.load(fileobj)
    with open(path, 'rb') as f:
        return pickle.load(f)


class ControlLogic:
    """
    Buffer last N predictions; emit class only when any class reaches threshold.
    """
    def __init__(self, buffer_size: int, threshold: int):
        self.buffer = deque(maxlen=buffer_size)
        self.threshold = threshold

    def update(self, pred: int):
        self.buffer.append(pred)
        if len(self.buffer) == self.buffer.maxlen:
            counts = {lab: self.buffer.count(lab) for lab in set(self.buffer)}
            top, cnt = max(counts.items(), key=lambda x: x[1])
            if cnt >= self.threshold:
                return top
        return None


# Number of channels after preprocessing (used for live PLV visualiser)
n_channels = len(Preprocessor().subset_indices)
