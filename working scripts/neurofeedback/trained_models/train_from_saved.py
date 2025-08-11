# train_from_saved.py
# Train CSP-LDA and PLV-GAT for ALL subjects' Session_001 under a given root.

import os
import pickle
import json
from glob import glob

import numpy as np
import scipy.signal as sig
from tqdm import tqdm

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric import seed_everything

# ============================
# USER CONFIG
# ============================

# >>>>>> SET THIS TO THE PARENT OF YOUR Subject_* FOLDERS <<<<<<
TRAINING_RESULTS_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\neurofeedback\training_results"

# Where to save models per subject/session
MODELS_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\neurofeedback\trained_models"

# Validation split (set to 0.0 to disable)
VAL_FRACTION        = 0.10   # e.g., 0.10 = 10% hold-out; 0.0 disables validation
RNG_SEED            = 12345

# Sampling rate fallback (we auto-read from files when present)
SFREQ = 256.0

# Windowing
WINDOW_SEC = 3.0
HOP_SEC    = 0.5

# PLV graph params (match online)
TOPK_PERCENT = 0.4
EPSILON      = 1e-6

# Pretrained GAT path (foundation)
PRETRAINED_GAT_PATH = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\neurofeedback\trained_models\foundational.pt"
FINETUNE_EPOCHS     = 15
BATCH_SIZE          = 32
LR                  = 1e-4
WEIGHT_DECAY        = 1e-4
DEVICE              = torch.device("cpu")

# Set seeds (best-effort reproducibility)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
seed_everything(RNG_SEED)

# ============================
# ELECTRODE MAPS
# ============================

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

CSP_CHANNELS = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
]

# ============================
# HELPERS
# ============================

def idx_map(src_names, keep_names):
    src_idx = {ch: i for i, ch in enumerate(src_names)}
    return [src_idx[ch] for ch in keep_names]

def segment_continuous(eeg_T: np.ndarray, sfreq: float, window_sec=3.0, hop_sec=0.5):
    n_ch, n_samp = eeg_T.shape
    win = int(round(window_sec * sfreq))
    hop = int(round(hop_sec * sfreq))
    if n_samp < win:
        return np.empty((0, win, n_ch))
    out = []
    for start in range(0, n_samp - win + 1, hop):
        end = start + win
        out.append(eeg_T[:, start:end].T)  # [win, n_ch]
    return np.stack(out, axis=0)

def compute_plv(seg):  # seg: [T, C]
    analytic = sig.hilbert(seg, axis=0)
    phase = np.angle(analytic)
    C = phase.shape[1]
    plv = np.eye(C, dtype=np.float32)
    for i in range(C):
        di = phase[:, i]
        for j in range(i+1, C):
            dj = phase[:, j]
            d  = dj - di
            val = np.abs(np.exp(1j*d).mean())
            plv[i, j] = plv[j, i] = val
    return plv

def plv_to_graph(plv, topk_percent=0.4, eps=1e-6):
    W = -np.log(1.0 - plv + eps).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    C = W.shape[0]
    triu = np.triu_indices(C, k=1)
    w = W[triu]
    k = max(1, int(round(len(w) * topk_percent)))
    top_idx = np.argpartition(w, -k)[-k:]
    rows, cols = triu[0][top_idx], triu[1][top_idx]
    ei = np.hstack([np.stack([rows, cols], axis=0), np.stack([cols, rows], axis=0)])
    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=C)
    return edge_index, torch.from_numpy(W)

def load_trials_one_session(session_pkl_path):
    if not os.path.isfile(session_pkl_path):
        return []
    with open(session_pkl_path, "rb") as f:
        d = pickle.load(f)
    trials = []
    for _, rec in d.items():
        eeg = rec.get('eeg', None)
        if eeg is None or eeg.size == 0:
            continue
        trials.append(rec)
    return trials

def assert_sfreq(trials, default_sfreq=SFREQ):
    fs_vals = {int(rec.get('fs', default_sfreq)) for rec in trials}
    if not fs_vals:
        print(f"⚠️ No 'fs' in trials; assuming SFREQ = {default_sfreq}")
        return float(default_sfreq)
    if len(fs_vals) > 1:
        raise RuntimeError(f"Inconsistent sampling rates in trials: {fs_vals}")
    return float(fs_vals.pop())

def stratified_split_indices(y: np.ndarray, val_fraction: float, seed: int = 42):
    """
    Returns (train_idx, val_idx). If val_fraction <= 0, val_idx is empty.
    Stratified by class labels in y (1D array-like).
    """
    y = np.asarray(y).ravel()
    n = len(y)
    if val_fraction <= 0.0 or n == 0:
        return np.arange(n, dtype=int), np.array([], dtype=int)

    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        # Hold out at least 1 if class has >= 2 samples
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) >= 2 else 0
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val_idx   = np.concatenate(val_idx)   if val_idx else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

# ============================
# CSP-LDA TRAIN
# ============================

def train_csp_lda_from_trials(trials, sfreq, window_sec=3.0, hop_sec=0.5, val_fraction=0.0, seed=42):
    idx_csp = idx_map(HEADSET_64, CSP_CHANNELS)
    X_list, y_list = [], []
    for rec in tqdm(trials, desc="CSP windows"):
        eeg_T = rec['eeg']
        if eeg_T.shape[0] != len(HEADSET_64):
            continue
        segs = segment_continuous(eeg_T[idx_csp, :], sfreq, window_sec, hop_sec)
        if segs.size == 0:
            continue
        X_list.append(segs)
        y_list.append(np.full((segs.shape[0],), int(rec['label']), dtype=int))

    if not X_list:
        raise RuntimeError("No CSP windows available for this session.")

    X = np.concatenate(X_list, axis=0)                     # (N, win, C)
    y = np.concatenate(y_list, axis=0)                     # (N,)
    X_mne = np.transpose(X, (0, 2, 1)).astype(np.float64)  # (N, C, T)
    y_use = (y - 1).astype(int) if set(np.unique(y)) == {1, 2} else y.astype(int)

    # --- stratified split ---
    tr_idx, va_idx = stratified_split_indices(y_use, val_fraction, seed)
    has_val = va_idx.size > 0

    X_tr, y_tr = X_mne[tr_idx], y_use[tr_idx]
    X_va, y_va = (X_mne[va_idx], y_use[va_idx]) if has_val else (None, None)

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    csp.fit(X_tr, y_tr)

    Xc_tr = csp.transform(X_tr)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xc_tr, y_tr)

    val_acc = None
    if has_val:
        Xc_va = csp.transform(X_va)
        y_pred = lda.predict(Xc_va)
        val_acc = float((y_pred == y_va).mean())
        print(f"   CSP-LDA val acc: {val_acc:.2%}  (N={len(y_va)})")
    else:
        print(f"   CSP-LDA trained with no validation split (VAL_FRACTION={val_fraction}).")

    return {
        "filters": csp.filters_,
        "lda_coef": lda.coef_,
        "lda_intercept": lda.intercept_,
        "csp_channels": CSP_CHANNELS,
        "sfreq": sfreq,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "val_fraction": float(val_fraction),
        "val_acc": val_acc,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "seed": int(seed),
    }

# ============================
# PLV-GAT FINETUNE
# ============================

class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, h1, heads=heads, concat=True, dropout=dropout)
        self.gn1   = GraphNorm(h1*heads)
        self.conv2 = GATv2Conv(h1*heads, h2, heads=heads, concat=True, dropout=dropout)
        self.gn2   = GraphNorm(h2*heads)
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

def build_plv_graphs_from_trials(trials, sfreq, window_sec=3.0, hop_sec=0.5, topk=0.4, eps=1e-6):
    idx_58 = idx_map(HEADSET_64, SHARED_58)
    graphs = []
    for rec in tqdm(trials, desc="PLV windows"):
        eeg_T = rec['eeg']
        if eeg_T.shape[0] != len(HEADSET_64):
            continue
        eeg58_T = eeg_T[idx_58, :]
        segs = segment_continuous(eeg58_T, sfreq, window_sec, hop_sec)
        if segs.size == 0:
            continue
        y = int(rec['label'])
        if y in (1, 2):
            y = y - 1
        for s in segs:
            plv = compute_plv(s)
            edge_index, W = plv_to_graph(plv, topk, eps)
            x = torch.eye(W.shape[0], dtype=torch.float32)  # identity node features
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
            graphs.append(data)
    if not graphs:
        raise RuntimeError("No PLV windows available for this session.")
    return graphs

def infer_gat_dims_from_state_dict(sd):
    heads = sd['conv1.att'].shape[1]
    h1    = sd['conv1.att'].shape[2]
    h2    = sd['conv2.att'].shape[2]
    h3    = sd['conv3.att'].shape[2]
    in_ch = sd['conv1.lin_l.weight'].shape[1]
    return in_ch, h1, h2, h3, heads

def finetune_gat(graphs, pretrained_path, out_path, epochs=15, batch_size=32, lr=1e-4, wd=1e-4,
                 device=torch.device("cpu"), val_fraction=0.0, seed=42):
    # --- split graphs stratified by label ---
    y_all = np.array([int(g.y.item()) for g in graphs], dtype=int)
    tr_idx, va_idx = stratified_split_indices(y_all, val_fraction, seed)
    has_val = va_idx.size > 0

    train_graphs = [graphs[i] for i in tr_idx]
    val_graphs   = [graphs[i] for i in va_idx] if has_val else []

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False) if has_val else None

    sd = torch.load(pretrained_path, map_location=device)
    in_ch, h1, h2, h3, heads = infer_gat_dims_from_state_dict(sd)

    model = SimpleGAT(in_ch, h1, h2, h3, heads).to(device)
    model.load_state_dict(sd)
    model.train()

    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    best_metric = 0.0
    best_based_on = "train" if not has_val else "val"

    def eval_loader(loader):
        if loader is None:
            return None
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total   += batch.num_graphs
        model.train()
        return (correct / max(1, total)) if total > 0 else 0.0

    for ep in range(1, epochs+1):
        # --- train ---
        correct = total = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss   = crit(logits, batch.y)
            loss.backward()
            opt.step()
            preds   = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += batch.num_graphs
        train_acc = correct / max(1, total)

        # --- validate (if any) ---
        val_acc = eval_loader(val_loader) if has_val else None
        disp = f"Epoch {ep}/{epochs}  Train: {train_acc:.2%}"
        if has_val:
            disp += f" | Val: {val_acc:.2%}"
        print(disp)

        # --- model selection ---
        metric_now = val_acc if has_val else train_acc
        if metric_now is not None and metric_now > best_metric:
            best_metric = metric_now
            torch.save(model.state_dict(), out_path)

    print(f"✅ GAT fine-tune done. Best {best_based_on} acc: {best_metric:.2%}. Saved → {out_path}")
    return best_metric, best_based_on, int(len(tr_idx)), int(len(va_idx))

# ============================
# DRIVER
# ============================

def main():
    root = os.path.abspath(TRAINING_RESULTS_ROOT)
    models_root = os.path.abspath(MODELS_ROOT)
    print(f"🔎 Looking for subjects under: {root}")
    if not os.path.isdir(root):
        raise RuntimeError(f"Training results root does not exist: {root}")

    # Case A: root contains Subject_* folders
    subject_dirs = sorted(glob(os.path.join(root, "Subject_*")))
    # Case B: root itself is a Subject_* folder
    if not subject_dirs and os.path.basename(root).startswith("Subject_"):
        subject_dirs = [root]

    # Fallback recursive (helps if you have nested structure)
    if not subject_dirs:
        print("⚠️ No Subject_* found directly; trying recursive search…")
        subject_dirs = sorted(glob(os.path.join(root, "**", "Subject_*"), recursive=True))

    if not subject_dirs:
        entries = os.listdir(root)
        raise RuntimeError(f"No subjects found under {root}\nRoot contents: {entries}")

    print("📁 Found subjects:")
    for p in subject_dirs:
        print("  -", p)

    for sdir in subject_dirs:
        subj = os.path.basename(sdir)
        sess_dir = os.path.join(sdir, "Session_001")
        session_pkl = os.path.join(sess_dir, "session_data.pkl")
        if not os.path.isfile(session_pkl):
            print(f"⏭️  Skipping {subj}: missing {session_pkl}")
            continue

        print(f"\n======================")
        print(f" Subject: {subj}")
        print(f" Session: Session_001")
        print(f"======================")

        trials = load_trials_one_session(session_pkl)
        if not trials:
            print(f"⏭️  No usable trials for {subj}/Session_001.")
            continue

        fs = assert_sfreq(trials, SFREQ)

        # Output dir
        out_dir = os.path.join(models_root, subj, "Session_001", "finetuned_models")
        os.makedirs(out_dir, exist_ok=True)
        csp_out  = os.path.join(out_dir, "csp_lda_static.pkl")
        gat_out  = os.path.join(out_dir, "gat_finetuned_from_nf.pt")
        meta_out = os.path.join(out_dir, "model_meta.json")

        # ---- CSP-LDA ----
        print("\nTraining CSP-LDA on motor-region channels…")
        try:
            csp_pack = train_csp_lda_from_trials(
                trials, fs, WINDOW_SEC, HOP_SEC,
                val_fraction=VAL_FRACTION, seed=RNG_SEED
            )
            with open(csp_out, "wb") as f:
                pickle.dump(csp_pack, f)
            print(f"✅ Saved CSP+LDA → {csp_out}")
            print(f"   Filters shape: {np.asarray(csp_pack['filters']).shape}")
            print(f"   LDA coef shape: {np.asarray(csp_pack['lda_coef']).shape}")
        except Exception as e:
            print(f"❌ CSP-LDA failed for {subj}/Session_001: {e}")
            csp_pack = None

        # ---- PLV-GAT ----
        print("\nBuilding PLV graphs (58-channel order) …")
        try:
            graphs = build_plv_graphs_from_trials(
                trials, fs, WINDOW_SEC, HOP_SEC, TOPK_PERCENT, EPSILON
            )
            print(f"Total graphs: {len(graphs)} (windows)")
            print("\nFine-tuning GAT on PLV graphs…")
            best_acc, best_on, n_tr, n_va = finetune_gat(
                graphs, PRETRAINED_GAT_PATH, gat_out,
                epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
                lr=LR, wd=WEIGHT_DECAY, device=DEVICE,
                val_fraction=VAL_FRACTION, seed=RNG_SEED
            )
        except Exception as e:
            print(f"❌ PLV-GAT fine-tune failed for {subj}/Session_001: {e}")
            best_acc, best_on, n_tr, n_va = None, None, None, None

        # ---- META ----
        meta = {
            "subject": subj,
            "session": "Session_001",
            "sfreq": fs,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "headset_64": HEADSET_64,
            "shared_58": SHARED_58,
            "csp_channels": CSP_CHANNELS,
            "topk_percent": TOPK_PERCENT,
            "epsilon": EPSILON,
            "val_fraction": VAL_FRACTION,
            "outputs": {
                "csp_lda": csp_out if csp_pack is not None else None,
                "gat_pt":  gat_out if best_acc is not None else None
            }
        }
        if csp_pack is not None:
            meta["csp"] = {
                "val_acc": csp_pack.get("val_acc", None),
                "n_train": csp_pack.get("n_train", None),
                "n_val":   csp_pack.get("n_val", None),
                "seed":    csp_pack.get("seed", None),
            }
        if best_acc is not None:
            meta["gat"] = {
                "best_metric": best_acc,
                "selected_on": best_on,   # "val" or "train"
                "n_train": n_tr,
                "n_val": n_va,
                "seed": RNG_SEED,
                "pretrained_gat_path": PRETRAINED_GAT_PATH
            }

        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"📝 Wrote meta → {meta_out}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
