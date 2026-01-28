# eval_finetune_gat_window_overlap_MATCH_FOUNDATION_NOSELFLOOPS_PLUS_CSP_PLV_LANDSCAPE.py
#
# PART 1: Fine-tune GAT (select best by VAL, report final TEST)
# PART 2: CSP + classical models (grid search on TRAIN only, pick best by VAL, report TEST)
#         + CSP topoplots via MNE EvokedArray.plot_topomap
#         + NEW: CSP can run on either ALL channels or a MOTOR subset (toggle below)
# PART 3: Feature landscape using PLV (CV vs symmetric KL on TRAIN ONLY)
#
# Split: per-class temporal split (by window time index t)

import os
import json
import pickle
from collections import Counter

import numpy as np
import scipy.signal as sig
from scipy.signal import butter, filtfilt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# =============================
# CONFIG
# =============================
SESSION_PKL = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\training_results\Subject_007\Session_001\session_data.pkl"
FOUNDATION_PT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\trained_models\3s_0.5s_8-30Hz_model.pt"
OUT_DIR = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\trained_models\Subject_007\Session_001_runs"

# --- Windowing ---
WINDOW_SEC          = 3.0
OVERLAP_SEC         = 0.5
KEEP_SHORT_AS_FULL  = True

# --- Preprocessing
APPLY_BANDPASS      = True
BP_LO, BP_HI        = 8.0, 30.0
BP_ORDER            = 4

# --- PLV transform ---
TOPK_PERCENT        = 0.40   # used for GAT edges only
EPSILON             = 1e-6

# --- GAT finetune ---
EPOCHS              = 100
BATCH_SIZE          = 32
LR                  = 1e-4
WEIGHT_DECAY        = 1e-4
DEVICE_STR          = 'cuda' if torch.cuda.is_available() else 'cpu'

SFREQ_FALLBACK      = 256.0
RNG_SEED            = 12345

# --- Split fractions (per-class temporal) ---
TRAIN_FRAC = 0.40
VAL_FRAC   = 0.30
TEST_FRAC  = 0.30

# --- CSP ---
CSP_NCOMP = 6
CSP_REG   = 1e-6

#  CSP channel selection toggle
CSP_MOTOR_SUBSET = True   # True -> CSP only on motor subset, False -> CSP on all 58

# Define motor subset in SHARED_58 naming (edit if you want tighter/looser set)
CSP_MOTOR_CHS = [
    "FC5","FC3","FC1","FCz","FC2","FC4","FC6",
    "C5","C3","C1","Cz","C2","C4","C6",
    "CP5","CP3","CP1","CPz","CP2","CP4","CP6",
    "P5","P3","P1","Pz","P2","P4","P6",
]

# --- PLV feature landscape ---
KL_NBINS = 20
KL_EPS   = 1e-12
CV_EPS   = 1e-10

# --- Part 3 selection thresholds ---
CV_KEEP_PCTL = 30
KL_KEEP_PCTL = 70


# ---------------------------
# CHANNEL LISTS (must match your saved order)
# ---------------------------
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


# =============================
# Reproducibility
# =============================
def set_seeds(seed=RNG_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        from torch_geometric import seed_everything
        seed_everything(seed)
    except Exception:
        pass


# =============================
# Data utils
# =============================
def idx_map(src_names, keep_names):
    src_idx = {ch: i for i, ch in enumerate(src_names)}
    missing = [ch for ch in keep_names if ch not in src_idx]
    if missing:
        raise RuntimeError(f"Missing channels in src list: {missing}")
    return [src_idx[ch] for ch in keep_names]

def load_trials_one_session(session_pkl_path):
    if not os.path.isfile(session_pkl_path):
        raise FileNotFoundError(f"Missing session file: {session_pkl_path}")
    with open(session_pkl_path, 'rb') as f:
        d = pickle.load(f)

    trials = []
    for _, rec in d.items():
        eeg = rec.get('eeg', None)
        if eeg is None or getattr(eeg, 'size', 0) == 0:
            continue
        trials.append(rec)

    if not trials:
        raise RuntimeError("No usable trials in session file.")
    return trials

def assert_sfreq(trials, default_sfreq=SFREQ_FALLBACK):
    fs_vals = {float(rec.get('fs', default_sfreq)) for rec in trials}
    if len(fs_vals) == 1:
        return float(next(iter(fs_vals)))
    if len(fs_vals) == 0:
        print(f"⚠️ No 'fs' present; assuming {default_sfreq} Hz")
        return float(default_sfreq)
    raise RuntimeError(f"Inconsistent sampling rates across trials: {fs_vals}")

def bandpass_continuous(x_CT: np.ndarray, fs: float, lo=BP_LO, hi=BP_HI, order=BP_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, x_CT, axis=1)

def segment_continuous(eeg_CT: np.ndarray, sfreq: float, window_sec: float, overlap_sec: float):
    if overlap_sec < 0:
        overlap_sec = 0.0
    if overlap_sec >= window_sec:
        overlap_sec = window_sec - (1.0 / max(1.0, sfreq))

    C, T = eeg_CT.shape
    win = int(round(window_sec * sfreq))
    ovl = int(round(overlap_sec * sfreq))
    hop = max(1, win - ovl)

    if T < win:
        return np.empty((0, win, C))

    out = []
    for start in range(0, T - win + 1, hop):
        end = start + win
        out.append(eeg_CT[:, start:end].T)  # -> [win, C]
    return np.stack(out, axis=0)

def fix_len_tc(seg_TC: np.ndarray, win: int):
    T, C = seg_TC.shape
    if T == win:
        return seg_TC.astype(np.float32)
    if T > win:
        return seg_TC[:win].astype(np.float32)
    reps = int(np.ceil(win / max(1, T)))
    tiled = np.tile(seg_TC, (reps, 1))
    return tiled[:win].astype(np.float32)

def compute_plv(seg_TC: np.ndarray):
    analytic = sig.hilbert(seg_TC, axis=0)
    phase = np.angle(analytic)
    C = phase.shape[1]
    plv = np.eye(C, dtype=np.float32)
    for i in range(C):
        di = phase[:, i]
        for j in range(i + 1, C):
            dj = phase[:, j]
            val = np.abs(np.exp(1j * (dj - di)).mean())
            plv[i, j] = plv[j, i] = val
    return plv

def plv_transform(plv_raw: np.ndarray, eps=EPSILON):
    plv = plv_raw.astype(np.float32).copy()
    np.fill_diagonal(plv, 0.0)
    X = -np.log(1.0 - plv + eps).astype(np.float32)
    np.fill_diagonal(X, 0.0)
    return X

def plv_to_graph_transformed_topk_noself(plv_raw: np.ndarray, topk_percent=TOPK_PERCENT, eps=EPSILON):
    X = plv_transform(plv_raw, eps=eps)
    C = X.shape[0]
    triu = np.triu_indices(C, k=1)
    w = X[triu]

    k = max(1, int(round(w.size * float(topk_percent))))
    top_idx = np.argpartition(w, -k)[-k:]

    rows = triu[0][top_idx]
    cols = triu[1][top_idx]

    ei = np.hstack([
        np.stack([rows, cols], axis=0),
        np.stack([cols, rows], axis=0),
    ])
    edge_index = torch.tensor(ei, dtype=torch.long)

    return Data(x=torch.from_numpy(X), edge_index=edge_index)


# =============================
# Temporal split train/val/test
# =============================
def temporal_split_per_class_train_val_test(graphs, y_all, train_frac, val_frac):
    y_all = np.asarray(y_all).ravel()
    classes = sorted(set(int(v) for v in y_all.tolist()))
    by_class = {}

    for c in classes:
        idx_c = np.where(y_all == c)[0]
        t_c = np.array([int(graphs[i].t) for i in idx_c])
        order = np.argsort(t_c, kind='stable')
        idx_sorted = idx_c[order]
        n_c = len(idx_sorted)

        n_tr = int(np.ceil(train_frac * n_c)) if n_c > 0 else 0
        n_va = int(np.ceil(val_frac   * n_c)) if n_c > 0 else 0

        tr = idx_sorted[:n_tr]
        va = idx_sorted[n_tr:n_tr+n_va]
        te = idx_sorted[n_tr+n_va:]

        by_class[c] = [list(tr), list(va), list(te)]

    for c in classes:
        if len(by_class[c][0]) == 0:
            if len(by_class[c][1]) > 0:
                by_class[c][0].append(by_class[c][1].pop(0))
            elif len(by_class[c][2]) > 0:
                by_class[c][0].append(by_class[c][2].pop(0))

    for c in classes:
        if len(by_class[c][1]) == 0 and len(by_class[c][2]) > 0:
            by_class[c][1].append(by_class[c][2].pop(0))

    train_idx = np.array([i for c in classes for i in by_class[c][0]], dtype=int)
    val_idx   = np.array([i for c in classes for i in by_class[c][1]], dtype=int)
    test_idx  = np.array([i for c in classes for i in by_class[c][2]], dtype=int)

    train_idx = train_idx[np.argsort([int(graphs[i].t) for i in train_idx])]
    val_idx   = val_idx[np.argsort([int(graphs[i].t) for i in val_idx])]
    test_idx  = test_idx[np.argsort([int(graphs[i].t) for i in test_idx])]
    return train_idx, val_idx, test_idx


# =============================
# GAT model
# =============================
class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch,    h1, heads=heads, concat=True,  dropout=dropout)
        self.gn1   = GraphNorm(h1 * heads)
        self.conv2 = GATv2Conv(h1*heads, h2, heads=heads, concat=True,  dropout=dropout)
        self.gn2   = GraphNorm(h2 * heads)
        self.conv3 = GATv2Conv(h2*heads, h3, heads=heads, concat=False, dropout=dropout)
        self.gn3   = GraphNorm(h3)
        self.lin   = nn.Linear(h3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x)

def infer_dims(sd):
    heads = sd['conv1.att'].shape[1]
    h1    = sd['conv1.att'].shape[2]
    h2    = sd['conv2.att'].shape[2]
    h3    = sd['conv3.att'].shape[2]
    in_ch = sd['conv1.lin_l.weight'].shape[1]
    return in_ch, h1, h2, h3, heads


# =============================
# Build graphs + window epochs
# =============================
def build_graphs(trials, sfreq, window_sec, overlap_sec, keep_short_as_full):
    idx_58 = idx_map(HEADSET_64, SHARED_58)
    graphs = []
    t_counter = 0
    win = int(round(window_sec * sfreq))

    for rec in tqdm(trials, desc="Building graphs (PLV->GAT)"):
        eeg = rec.get('eeg', None)
        if eeg is None or eeg.shape[0] != len(HEADSET_64):
            continue

        y = int(rec.get('label'))
        if y not in (0, 1):
            raise ValueError(f"Unexpected label {y}; expected 0/1.")

        eeg58 = eeg[idx_58, :]

        if APPLY_BANDPASS:
            eeg58 = bandpass_continuous(eeg58, sfreq)

        segs = segment_continuous(eeg58, sfreq, window_sec, overlap_sec)
        if segs.size == 0 and keep_short_as_full:
            s = fix_len_tc(eeg58.T, win)
            segs = np.expand_dims(s, axis=0)

        for s in segs:
            if s.shape[0] != win:
                s = fix_len_tc(s, win)
            plv = compute_plv(s)
            data = plv_to_graph_transformed_topk_noself(plv, TOPK_PERCENT, EPSILON)
            data.y = torch.tensor([y], dtype=torch.long)
            data.t = int(t_counter)
            t_counter += 1
            graphs.append(data)

    if not graphs:
        raise RuntimeError("No graphs built.")
    return graphs

def build_window_epochs(trials, sfreq, window_sec, overlap_sec, keep_short_as_full):
    idx_58 = idx_map(HEADSET_64, SHARED_58)
    X_list, y_list, t_list = [], [], []
    t_counter = 0
    win = int(round(window_sec * sfreq))

    for rec in tqdm(trials, desc="Building window epochs (for CSP/PLV landscape)"):
        eeg = rec.get('eeg', None)
        if eeg is None or eeg.shape[0] != len(HEADSET_64):
            continue

        y = int(rec.get('label'))
        if y not in (0, 1):
            raise ValueError(f"Unexpected label {y}; expected 0/1.")

        eeg58 = eeg[idx_58, :]

        if APPLY_BANDPASS:
            eeg58 = bandpass_continuous(eeg58, sfreq)

        segs = segment_continuous(eeg58, sfreq, window_sec, overlap_sec)
        if segs.size == 0 and keep_short_as_full:
            s = fix_len_tc(eeg58.T, win)
            segs = np.expand_dims(s, axis=0)

        for s in segs:
            if s.shape[0] != win:
                s = fix_len_tc(s, win)
            X_list.append(s.T.astype(np.float32))  # [58, win]
            y_list.append(y)
            t_list.append(int(t_counter))
            t_counter += 1

    if not X_list:
        raise RuntimeError("No window epochs built.")
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=int), np.asarray(t_list, dtype=int)


# =============================
# Eval helpers
# =============================
def evaluate_gat(model, loader, device):
    model.eval()
    correct = total = 0
    cm = np.zeros((2,2), dtype=int)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += batch.num_graphs
            for t, p in zip(batch.y.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
    return (correct / max(1,total)), cm

def class_weight_tensor(graphs, device, num_classes: int = 2):
    y = np.array([int(g.y.item()) for g in graphs])
    counts = Counter(y)
    N = len(y)
    weights = []
    for c in range(num_classes):
        cnt = int(counts.get(c, 0))
        weights.append(0.0 if cnt == 0 else (N / (num_classes * cnt)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =============================
# PART 2: CSP
# =============================
def _cov_trial(X_CT, reg=CSP_REG):
    X = X_CT - X_CT.mean(axis=1, keepdims=True)
    cov = (X @ X.T) / max(1, X.shape[1] - 1)
    cov = cov / (np.trace(cov) + 1e-12)
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov

def fit_csp(X_train, y_train, n_components=CSP_NCOMP, reg=CSP_REG):
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    if len(X0) == 0 or len(X1) == 0:
        raise RuntimeError("CSP needs both classes in training set.")

    R0 = np.mean([_cov_trial(x, reg) for x in X0], axis=0)
    R1 = np.mean([_cov_trial(x, reg) for x in X1], axis=0)
    R  = R0 + R1

    evals, evecs = np.linalg.eigh(np.linalg.solve(R, R1))
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    W = evecs.astype(np.float32)
    A = np.linalg.pinv(W).T.astype(np.float32)

    k = int(n_components)
    k1 = k // 2
    k2 = k - k1
    picks = np.concatenate([np.arange(k2), np.arange(W.shape[1] - k1, W.shape[1])])
    return W, A, picks

def csp_transform(X, W, picks):
    feats = []
    for x in X:
        Z = (W.T @ x)
        Zp = Z[picks, :]
        var = np.var(Zp, axis=1)
        var = var / (np.sum(var) + 1e-12)
        feats.append(np.log(var + 1e-12))
    return np.stack(feats, axis=0).astype(np.float32)

def save_csp_topoplots_mne(A_patterns, picks, out_dir, prefix="CSP_topomap_patterns", n_plot=None, ch_names=None):
    """
    Plot CSP spatial patterns (A) as scalp topomaps.
    IMPORTANT: ch_names must correspond to A_patterns rows.
    """
    try:
        import mne
    except Exception as e:
        raise RuntimeError(
            "CSP topoplots require mne. Install with: pip install mne\n"
            f"Original import error: {e}"
        )

    os.makedirs(out_dir, exist_ok=True)
    if ch_names is None:
        ch_names = SHARED_58

    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="warn")

    comps = list(picks) if n_plot is None else list(picks[:int(n_plot)])

    for i, comp in enumerate(comps, start=1):
        vals = A_patterns[:, comp].astype(np.float64)
        evoked = mne.EvokedArray(vals[:, None], info, tmin=0.0)

        fig = evoked.plot_topomap(
            times=[0.0],
            scalings=1.0,
            time_format="",
            show=False,
            contours=0
        )
        if isinstance(fig, list):
            fig = fig[0]

        fp = os.path.join(out_dir, f"{prefix}_rank{i}_comp{comp}.png")
        fig.savefig(fp, dpi=250, bbox_inches="tight")
        plt.close(fig)


# =============================
# Classical model zoo + VAL selection
# =============================
def classical_model_zoo():
    zoo = {}
    zoo["logreg"] = (LogisticRegression(max_iter=2000, solver="liblinear"),
                    {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"]})
    zoo["svm_linear"] = (SVC(kernel="linear"),
                        {"clf__C": [0.01, 0.1, 1.0, 10.0]})
    zoo["svm_rbf"] = (SVC(kernel="rbf"),
                     {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1, 1.0]})
    zoo["knn"] = (KNeighborsClassifier(),
                 {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]})
    zoo["rf"] = (RandomForestClassifier(),
                {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10]})
    zoo["gboost"] = (GradientBoostingClassifier(),
                    {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]})
    return zoo

def fit_grid_on_train_select_by_val(Xtr, ytr, Xva, yva, Xte, yte, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    best = {"name": None, "val_acc": -1.0, "test_acc": None, "val_cm": None, "test_cm": None, "best_params": None}

    for name, (base_clf, grid) in classical_model_zoo().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])

        gs = GridSearchCV(pipe, grid, scoring="accuracy", cv=5, n_jobs=-1)
        gs.fit(Xtr, ytr)

        yhat_va = gs.best_estimator_.predict(Xva)
        va_acc = accuracy_score(yva, yhat_va)
        va_cm  = confusion_matrix(yva, yhat_va, labels=[0,1]).astype(int)

        yhat_te = gs.best_estimator_.predict(Xte)
        te_acc = accuracy_score(yte, yhat_te)
        te_cm  = confusion_matrix(yte, yhat_te, labels=[0,1]).astype(int)

        all_results.append({
            "model": name,
            "val_acc": float(va_acc),
            "val_cm": va_cm.tolist(),
            "test_acc": float(te_acc),
            "test_cm": te_cm.tolist(),
            "best_params": gs.best_params_
        })

        if va_acc > best["val_acc"]:
            best.update({
                "name": name,
                "val_acc": float(va_acc),
                "test_acc": float(te_acc),
                "val_cm": va_cm.tolist(),
                "test_cm": te_cm.tolist(),
                "best_params": gs.best_params_
            })

    summ_path = os.path.join(out_dir, f"{tag}_classical_summary.json")
    with open(summ_path, "w") as f:
        json.dump({"tag": tag, "all": all_results, "best": best}, f, indent=2)

    return best, all_results


# =============================
# PART 3: PLV feature landscape (CV vs KL)
# =============================
def compute_cv(vec, eps=CV_EPS):
    mu = float(np.mean(vec))
    sd = float(np.std(vec))
    denom = abs(mu) + eps
    if denom < 1e-20:
        return 0.0
    return float(sd / denom)

def sym_kl_hist(a, b, nbins=KL_NBINS, eps=KL_EPS):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    bins = np.linspace(lo, hi, nbins + 1)
    pa, _ = np.histogram(a, bins=bins, density=False)
    pb, _ = np.histogram(b, bins=bins, density=False)

    pa = pa.astype(np.float64) + eps
    pb = pb.astype(np.float64) + eps
    pa /= pa.sum()
    pb /= pb.sum()

    kl_ab = np.sum(pa * np.log(pa / pb))
    kl_ba = np.sum(pb * np.log(pb / pa))
    return float(0.5 * (kl_ab + kl_ba))

def plv_features_from_epoch(x_ct):
    seg_tc = x_ct.T
    plv = compute_plv(seg_tc)
    X = plv_transform(plv, eps=EPSILON)
    C = X.shape[0]
    triu = np.triu_indices(C, k=1)
    edges = X[triu].astype(np.float32)
    nodes = np.sum(np.abs(X), axis=1).astype(np.float32)
    return edges, nodes

def compute_cv_kl_landscape(F_edges, F_nodes, y):
    y = np.asarray(y).astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise RuntimeError("Need both classes to compute CV/KL landscape.")

    def family_stats(F):
        nfeat = F.shape[1]
        cvs = np.zeros(nfeat, dtype=np.float32)
        kls = np.zeros(nfeat, dtype=np.float32)
        for u in range(nfeat):
            v0 = F[idx0, u]
            v1 = F[idx1, u]
            cvs[u] = 0.5 * (compute_cv(v0) + compute_cv(v1))
            kls[u] = sym_kl_hist(v0, v1)
        return cvs, kls

    cv_e, kl_e = family_stats(F_edges)
    cv_n, kl_n = family_stats(F_nodes)
    return cv_e, kl_e, cv_n, kl_n

def plot_landscape(cv_e, kl_e, cv_n, kl_n, out_dir, tag="PLV"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.scatter(cv_e, kl_e, s=8)
    plt.xlabel("CV (lower = more stable)")
    plt.ylabel("Symmetric KL (higher = more discriminative)")
    plt.title(f"{tag} Edge Feature Landscape (TRAIN)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_landscape_edges_CV_vs_KL.png"), dpi=250)
    plt.close()

    plt.figure()
    plt.scatter(cv_n, kl_n, s=20)
    plt.xlabel("CV (lower = more stable)")
    plt.ylabel("Symmetric KL (higher = more discriminative)")
    plt.title(f"{tag} Node Feature Landscape (TRAIN)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_landscape_nodes_CV_vs_KL.png"), dpi=250)
    plt.close()

    plt.figure()
    plt.scatter(cv_e, kl_e, s=8, label="edges")
    plt.scatter(cv_n, kl_n, s=20, label="nodes")
    plt.xlabel("CV (lower = more stable)")
    plt.ylabel("Symmetric KL (higher = more discriminative)")
    plt.title(f"{tag} Feature Landscape (TRAIN): edges + nodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_landscape_combined_CV_vs_KL.png"), dpi=250)
    plt.close()

def select_features_by_cv_kl(cv, kl, cv_keep_pctl=30, kl_keep_pctl=70):
    cv_thr = float(np.percentile(cv, cv_keep_pctl))
    kl_thr = float(np.percentile(kl, kl_keep_pctl))
    sel = np.where((cv <= cv_thr) & (kl >= kl_thr))[0]
    return sel.astype(int), cv_thr, kl_thr

def apply_feature_mask(F_edges, F_nodes, sel_e, sel_n):
    Fe = F_edges[:, sel_e] if sel_e.size > 0 else np.zeros((F_edges.shape[0], 0), dtype=np.float32)
    Fn = F_nodes[:, sel_n] if sel_n.size > 0 else np.zeros((F_nodes.shape[0], 0), dtype=np.float32)
    return np.concatenate([Fe, Fn], axis=1).astype(np.float32)


# =============================
# Run pipeline
# =============================
def run_pipeline(trials, sfreq, sd_path, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Build graphs for GAT
    # -------------------------
    graphs = build_graphs(trials, sfreq, WINDOW_SEC, OVERLAP_SEC, KEEP_SHORT_AS_FULL)
    y_all = np.array([int(g.y.item()) for g in graphs], dtype=int)
    counts = dict(Counter(y_all))

    hop_sec = WINDOW_SEC - OVERLAP_SEC
    print(f"\nSegmentation → window={WINDOW_SEC:.3f}s, overlap={OVERLAP_SEC:.3f}s, hop={hop_sec:.3f}s")
    print(f"Built {len(graphs)} graphs. Class counts: {counts}")
    print(f"Graph x shape: {tuple(graphs[0].x.shape)} (expect [58,58])")

    sd = torch.load(sd_path, map_location=device)
    in_ch, h1, h2, h3, heads = infer_dims(sd)

    x_in = graphs[0].x.shape[1]
    if int(x_in) != int(in_ch):
        raise RuntimeError(f"Feature dim mismatch: model expects in_ch={in_ch}, but graph has x.shape[1]={x_in}.")

    # Foundation eval
    full_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    model = SimpleGAT(in_ch, h1, h2, h3, heads).to(device)
    model.load_state_dict(sd)
    foundation_acc, foundation_cm = evaluate_gat(model, full_loader, device)
    print(f"Foundation — Acc (all windows): {foundation_acc:.2%}, CM=\n{foundation_cm}")

    # -------------------------
    # Train/Val/Test split (temporal per class)
    # -------------------------
    tr_idx, va_idx, te_idx = temporal_split_per_class_train_val_test(graphs, y_all, TRAIN_FRAC, VAL_FRAC)
    train_graphs = [graphs[i] for i in tr_idx]
    val_graphs   = [graphs[i] for i in va_idx]
    test_graphs  = [graphs[i] for i in te_idx]

    train_counts = Counter([int(g.y.item()) for g in train_graphs])
    val_counts   = Counter([int(g.y.item()) for g in val_graphs])
    test_counts  = Counter([int(g.y.item()) for g in test_graphs])

    print(f"\nSplit counts:")
    print(f"  Train: {len(train_graphs)} {dict(train_counts)}")
    print(f"  Val:   {len(val_graphs)} {dict(val_counts)}")
    print(f"  Test:  {len(test_graphs)} {dict(test_counts)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

    # =============================
    # PART 1 — Fine-tune GAT
    # =============================
    model_ft = SimpleGAT(in_ch, h1, h2, h3, heads).to(device)
    model_ft.load_state_dict(sd)
    model_ft.train()

    if len(train_counts.keys()) < 2:
        crit = nn.CrossEntropyLoss()
    else:
        class_w = class_weight_tensor(train_graphs, device, num_classes=2)
        crit = nn.CrossEntropyLoss(weight=class_w) if (class_w > 0).all() else nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model_ft.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_sd = None
    best_val_cm = None

    for ep in range(1, EPOCHS + 1):
        correct = total = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model_ft(batch)
            loss = crit(logits, batch.y)
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += batch.num_graphs

        tr_acc = correct / max(1, total)
        va_acc, va_cm = evaluate_gat(model_ft, val_loader, device)
        print(f"[GAT] Epoch {ep}/{EPOCHS} — Train {tr_acc:.2%} | Val {va_acc:.2%}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_sd = {k: v.detach().cpu().clone() for k, v in model_ft.state_dict().items()}
            best_val_cm = va_cm

    model_best = SimpleGAT(in_ch, h1, h2, h3, heads).to(device)
    model_best.load_state_dict(best_sd if best_sd is not None else model_ft.state_dict())
    test_acc_gat, test_cm_gat = evaluate_gat(model_best, test_loader, device)

    pt_path = os.path.join(out_dir, f"gat_finetuned_MATCHFOUND_noself_win{WINDOW_SEC:g}_ov{OVERLAP_SEC:.2f}s.pt")
    torch.save(model_best.state_dict(), pt_path)
    print(f"\n✅ Saved best (by VAL) GAT → {pt_path}")
    print(f"[GAT] Best VAL acc: {best_val_acc:.2%}, VAL CM=\n{best_val_cm}")
    print(f"[GAT] TEST acc:     {test_acc_gat:.2%}, TEST CM=\n{test_cm_gat}")

    # =============================
    # Build epochs (aligned) for CSP + PLV landscape
    # =============================
    X_epochs, y_epochs, _t_epochs = build_window_epochs(trials, sfreq, WINDOW_SEC, OVERLAP_SEC, KEEP_SHORT_AS_FULL)
    if len(X_epochs) != len(graphs):
        raise RuntimeError(f"Epoch/Graph alignment mismatch: epochs={len(X_epochs)} vs graphs={len(graphs)}.")

    Xtr = X_epochs[tr_idx]; ytr = y_epochs[tr_idx]
    Xva = X_epochs[va_idx]; yva = y_epochs[va_idx]
    Xte = X_epochs[te_idx]; yte = y_epochs[te_idx]

    # =============================
    # PART 2 — CSP + classical models + topoplots
    # =============================
    print("\n=============================")
    print("PART 2 — CSP + Classical Models")
    print("=============================")

    # NEW: choose CSP channels
    if CSP_MOTOR_SUBSET:
        csp_chs = CSP_MOTOR_CHS
        print(f"[PART 2] CSP channels: MOTOR subset ({len(csp_chs)} chs)")
    else:
        csp_chs = SHARED_58
        print(f"[PART 2] CSP channels: ALL ({len(csp_chs)} chs)")

    csp_idx = np.array(idx_map(SHARED_58, csp_chs), dtype=int)

    # Slice epochs to CSP channels only: X: [N, C, T] -> [N, C_sub, T]
    Xtr_csp = Xtr[:, csp_idx, :]
    Xva_csp = Xva[:, csp_idx, :]
    Xte_csp = Xte[:, csp_idx, :]

    W_csp, A_csp, picks_csp = fit_csp(Xtr_csp, ytr, n_components=CSP_NCOMP, reg=CSP_REG)
    Ftr_csp = csp_transform(Xtr_csp, W_csp, picks_csp)
    Fva_csp = csp_transform(Xva_csp, W_csp, picks_csp)
    Fte_csp = csp_transform(Xte_csp, W_csp, picks_csp)

    # Topos now match the CSP channel set (motor subset plots will be motor-only montage)
    topo_prefix = "CSP_topomap_patterns_MOTOR" if CSP_MOTOR_SUBSET else "CSP_topomap_patterns_ALL"
    save_csp_topoplots_mne(A_csp, picks_csp, out_dir, prefix=topo_prefix, n_plot=CSP_NCOMP, ch_names=csp_chs)

    best_csp, _ = fit_grid_on_train_select_by_val(
        Ftr_csp, ytr, Fva_csp, yva, Fte_csp, yte,
        out_dir=out_dir,
        tag="PART2_CSP_MOTOR" if CSP_MOTOR_SUBSET else "PART2_CSP_ALL"
    )
    print(f"[PART 2] Best-by-VAL: {best_csp['name']} | Val {best_csp['val_acc']:.2%} | Test {best_csp['test_acc']:.2%}")

    # =============================
    # PART 3 — PLV CV/KL selection + classical models
    # =============================
    print("\n=============================")
    print("PART 3 — PLV + CV/KL Feature Selection + Classical Models")
    print("=============================")

    edges_list, nodes_list = [], []
    for x_ct in tqdm(X_epochs, desc="Computing PLV features per window"):
        e, n = plv_features_from_epoch(x_ct)
        edges_list.append(e)
        nodes_list.append(n)

    F_edges = np.stack(edges_list, axis=0).astype(np.float32)  # [N, E]
    F_nodes = np.stack(nodes_list, axis=0).astype(np.float32)  # [N, 58]

    # Split feature matrices
    F_edges_tr, F_edges_va, F_edges_te = F_edges[tr_idx], F_edges[va_idx], F_edges[te_idx]
    F_nodes_tr, F_nodes_va, F_nodes_te = F_nodes[tr_idx], F_nodes[va_idx], F_nodes[te_idx]

    # TRAIN-only CV/KL
    cv_e, kl_e, cv_n, kl_n = compute_cv_kl_landscape(F_edges_tr, F_nodes_tr, ytr)

    # Plot landscapes (TRAIN)
    plot_landscape(cv_e, kl_e, cv_n, kl_n, out_dir, tag="PLV")

    # Select features (TRAIN-only thresholds)
    sel_e, cv_thr_e, kl_thr_e = select_features_by_cv_kl(cv_e, kl_e, CV_KEEP_PCTL, KL_KEEP_PCTL)
    sel_n, cv_thr_n, kl_thr_n = select_features_by_cv_kl(cv_n, kl_n, CV_KEEP_PCTL, KL_KEEP_PCTL)

    print(f"[PART 3] Edge feats: total={F_edges.shape[1]}, selected={sel_e.size} (CV<=p{CV_KEEP_PCTL}:{cv_thr_e:.4f}, KL>=p{KL_KEEP_PCTL}:{kl_thr_e:.4f})")
    print(f"[PART 3] Node feats: total={F_nodes.shape[1]}, selected={sel_n.size} (CV<=p{CV_KEEP_PCTL}:{cv_thr_n:.4f}, KL>=p{KL_KEEP_PCTL}:{kl_thr_n:.4f})")

    Xtr_sel = apply_feature_mask(F_edges_tr, F_nodes_tr, sel_e, sel_n)
    Xva_sel = apply_feature_mask(F_edges_va, F_nodes_va, sel_e, sel_n)
    Xte_sel = apply_feature_mask(F_edges_te, F_nodes_te, sel_e, sel_n)

    if Xtr_sel.shape[1] == 0:
        raise RuntimeError("PART 3 selected 0 features. Loosen thresholds or inspect the landscape.")

    fs_path = os.path.join(out_dir, "PART3_PLV_selected_features_CVKL.npz")
    np.savez(
        fs_path,
        sel_edge_idx=sel_e,
        sel_node_idx=sel_n,
        cv_edges=cv_e, kl_edges=kl_e,
        cv_nodes=cv_n, kl_nodes=kl_n,
        cv_thr_edges=cv_thr_e, kl_thr_edges=kl_thr_e,
        cv_thr_nodes=cv_thr_n, kl_thr_nodes=kl_thr_n,
        cv_keep_pctl=int(CV_KEEP_PCTL), kl_keep_pctl=int(KL_KEEP_PCTL),
        tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx
    )
    print(f"[PART 3] Saved selection + stats → {fs_path}")

    best_plv_sel, _ = fit_grid_on_train_select_by_val(
        Xtr_sel, ytr, Xva_sel, yva, Xte_sel, yte,
        out_dir=out_dir,
        tag="PART3_PLV_CVKL"
    )
    print(f"[PART 3] Best-by-VAL: {best_plv_sel['name']} | Val {best_plv_sel['val_acc']:.2%} | Test {best_plv_sel['test_acc']:.2%}")

    results = {
        "window_sec": float(WINDOW_SEC),
        "overlap_sec": float(OVERLAP_SEC),
        "hop_sec": float(WINDOW_SEC - OVERLAP_SEC),
        "apply_bandpass": bool(APPLY_BANDPASS),
        "bp_lo": float(BP_LO),
        "bp_hi": float(BP_HI),
        "topk_percent": float(TOPK_PERCENT),
        "epsilon": float(EPSILON),

        "split_fracs": {"train": float(TRAIN_FRAC), "val": float(VAL_FRAC), "test": float(TEST_FRAC)},
        "n_windows": int(len(graphs)),
        "train_class_counts": dict(train_counts),
        "val_class_counts": dict(val_counts),
        "test_class_counts": dict(test_counts),

        "foundation_acc_all": float(foundation_acc),
        "foundation_cm_all": foundation_cm.tolist(),

        "part1_gat_best_val_acc": float(best_val_acc),
        "part1_gat_best_val_cm": best_val_cm.tolist() if best_val_cm is not None else None,
        "part1_gat_test_acc": float(test_acc_gat),
        "part1_gat_test_cm": test_cm_gat.tolist(),
        "part1_gat_saved_to": pt_path,

        # PART 2 (now includes channel mode)
        "part2_csp_mode": "motor_subset" if CSP_MOTOR_SUBSET else "all",
        "part2_csp_channels": csp_chs,
        "part2_csp_best_model": best_csp["name"],
        "part2_csp_best_val_acc": float(best_csp["val_acc"]),
        "part2_csp_best_test_acc": float(best_csp["test_acc"]),
        "part2_csp_best_val_cm": best_csp["val_cm"],
        "part2_csp_best_test_cm": best_csp["test_cm"],
        "part2_csp_best_params": best_csp["best_params"],
        "part2_csp_topos_prefix": f"{topo_prefix}_rank*_comp*.png",

        # PART 3
        "part3_plv_feature_selection_npz": fs_path,
        "part3_plv_selected_edge_count": int(sel_e.size),
        "part3_plv_selected_node_count": int(sel_n.size),
        "part3_plv_cv_thr_edges": float(cv_thr_e),
        "part3_plv_kl_thr_edges": float(kl_thr_e),
        "part3_plv_cv_thr_nodes": float(cv_thr_n),
        "part3_plv_kl_thr_nodes": float(kl_thr_n),

        "part3_plv_best_model": best_plv_sel["name"],
        "part3_plv_best_val_acc": float(best_plv_sel["val_acc"]),
        "part3_plv_best_test_acc": float(best_plv_sel["test_acc"]),
        "part3_plv_best_val_cm": best_plv_sel["val_cm"],
        "part3_plv_best_test_cm": best_plv_sel["test_cm"],
        "part3_plv_best_params": best_plv_sel["best_params"],

        "part3_plv_landscape_plots": [
            os.path.join(out_dir, "PLV_landscape_edges_CV_vs_KL.png"),
            os.path.join(out_dir, "PLV_landscape_nodes_CV_vs_KL.png"),
            os.path.join(out_dir, "PLV_landscape_combined_CV_vs_KL.png"),
        ],
    }
    return results


def main():
    if not os.path.isfile(SESSION_PKL):
        raise FileNotFoundError(f"SESSION_PKL not found: {SESSION_PKL}")
    if not os.path.isfile(FOUNDATION_PT):
        raise FileNotFoundError(f"FOUNDATION_PT not found: {FOUNDATION_PT}")

    os.makedirs(OUT_DIR, exist_ok=True)
    set_seeds(RNG_SEED)

    trials = load_trials_one_session(SESSION_PKL)
    sfreq  = assert_sfreq(trials, SFREQ_FALLBACK)
    device = torch.device(DEVICE_STR)
    print(f"Device: {device}, sfreq: {sfreq}")

    results = run_pipeline(trials, sfreq, FOUNDATION_PT, OUT_DIR, device)

    report = {
        "session_pkl": SESSION_PKL,
        "foundation_pt": FOUNDATION_PT,
        "sfreq": float(sfreq),
        "results": results,
    }
    report_path = os.path.join(OUT_DIR, "report_MATCHFOUNDATION_noself_plus_CSP_PLV_landscape.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📝 Wrote report → {report_path}")


if __name__ == "__main__":
    main()
