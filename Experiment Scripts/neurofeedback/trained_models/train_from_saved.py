# eval_finetune_gat_window_overlap.py
# Self-contained Spyder-friendly script (OVERLAP IN SECONDS).
#   • Build PLV-graph dataset using one segmentation: WINDOW_SEC + OVERLAP_SEC
#   • Optionally keep short trials as single full segments
#   • Evaluate your foundation GAT (no fine-tune)
#   • Fine-tune on FIRST K% IN TIME PER CLASS (no balancing; class weights), then test on the rest
#   • Save best fine-tuned model and a summary report.json

import os
import json
import pickle
from collections import Counter

import numpy as np
import scipy.signal as sig
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops

# =============================
# CONFIG — EDIT THESE IN SPYDER
# =============================
SESSION_PKL   = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\training_results\Subject_000\Session_005\session_data.pkl"
FOUNDATION_PT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\trained_models\foundational.pt"
OUT_DIR       = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\trained_models\Session_005_runs"

# --- Windowing (single scheme) ---
WINDOW_SEC       = 3.0     # e.g., 1.0, 5.0, etc.
OVERLAP_SEC      = 0.5     # overlap IN SECONDS with the next window, must be < WINDOW_SEC
KEEP_SHORT_AS_FULL = True  # if trial < window, keep it as a single full segment

# Fine-tune & training params
FINETUNE_FRAC       = 0.50  # First K% in time PER CLASS for fine-tune; rest for test
EPOCHS              = 100
BATCH_SIZE          = 32
LR                  = 1e-4
WEIGHT_DECAY        = 1e-4
DEVICE_STR          = 'cuda' if torch.cuda.is_available() else 'cpu'  # override to 'cpu' if needed

# PLV graph construction
TOPK_PERCENT   = 0.40
EPSILON        = 1e-6
SFREQ_FALLBACK = 256.0
RNG_SEED       = 12345

# Channels
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

def segment_continuous(eeg_T: np.ndarray, sfreq: float, window_sec: float, overlap_sec: float):
    """
    Sliding windows with given overlap (in seconds).
    hop_sec = max(1 sample, window_sec - overlap_sec).
    Returns (segments_array, hop_sec).
      - segments_array: shape (N, win, n_ch) or empty array if trial < window.
      - hop_sec: float, effective hop in seconds.
    """
    # sanitize overlap
    if overlap_sec < 0:
        overlap_sec = 0.0
    if overlap_sec >= window_sec:
        # just smaller than window to avoid zero/negative hop
        overlap_sec = window_sec - (1.0 / max(1.0, sfreq))

    n_ch, n_samp = eeg_T.shape
    win = int(round(window_sec * sfreq))
    ovl = int(round(overlap_sec * sfreq))
    hop = max(1, win - ovl)
    hop_sec = hop / sfreq

    if n_samp < win:
        return np.empty((0, win, n_ch)), hop_sec

    out = []
    for start in range(0, n_samp - win + 1, hop):
        end = start + win
        out.append(eeg_T[:, start:end].T)  # [win, n_ch]
    return np.stack(out, axis=0), hop_sec

def compute_plv(seg: np.ndarray):  # seg: [T, C]
    analytic = sig.hilbert(seg, axis=0)
    phase = np.angle(analytic)
    C = phase.shape[1]
    plv = np.eye(C, dtype=np.float32)
    for i in range(C):
        di = phase[:, i]
        for j in range(i + 1, C):
            dj = phase[:, j]
            d  = dj - di
            val = np.abs(np.exp(1j * d).mean())
            plv[i, j] = plv[j, i] = val
    return plv

def plv_to_graph(plv: np.ndarray, topk_percent=TOPK_PERCENT, eps=EPSILON):
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
    # Node features = PLV rows (C, C)
    x = torch.from_numpy(plv.astype(np.float32))
    return Data(x=x, edge_index=edge_index)

# =============================
# Temporal, per-class split (no balancing; use class weights)
# =============================
def temporal_split_first_frac_per_class(graphs, y_all, frac: float):
    """
    Returns (train_idx, test_idx) using temporal order .t per class.
    - For each class c, sort indices by data.t and take first ceil(frac * n_c) as train.
    - If a class ends up empty in train, move earliest item from test→train.
    """
    y_all = np.asarray(y_all).ravel()
    classes = sorted(set(int(v) for v in y_all.tolist()))
    by_class = {}

    for c in classes:
        idx_c = np.where(y_all == c)[0]
        t_c = np.array([int(graphs[i].t) for i in idx_c])
        order = np.argsort(t_c, kind='stable')
        idx_sorted = idx_c[order]
        n_c = len(idx_sorted)
        n_train_c = int(np.ceil(frac * n_c)) if n_c > 0 else 0
        train_c = idx_sorted[:n_train_c]
        test_c  = idx_sorted[n_train_c:]
        by_class[c] = [list(train_c), list(test_c)]

    for c in classes:
        if len(by_class[c][0]) == 0 and len(by_class[c][1]) > 0:
            by_class[c][0].append(by_class[c][1].pop(0))

    train_idx = np.array([i for c in classes for i in by_class[c][0]], dtype=int)
    test_idx  = np.array([i for c in classes for i in by_class[c][1]], dtype=int)

    train_idx = train_idx[np.argsort([int(graphs[i].t) for i in train_idx])]
    test_idx  = test_idx[np.argsort([int(graphs[i].t) for i in test_idx])]
    return train_idx, test_idx

# =============================
# Model
# =============================
class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch,   h1, heads=heads, concat=True,  dropout=dropout)
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
# Build graphs (single window/overlap scheme)
# =============================
def build_graphs(trials, sfreq, window_sec: float, overlap_sec: float, keep_short_as_full: bool):
    """
    Returns list[Data] with:
      .x (C,C), .edge_index, .y in {0,1}, .t (temporal order)
    """
    idx_58 = idx_map(HEADSET_64, SHARED_58)
    graphs = []
    t_counter = 0
    for rec in tqdm(trials, desc="Building graphs"):
        eeg_T = rec['eeg']               # [64, T]
        if eeg_T.shape[0] != len(HEADSET_64):
            continue
        y = int(rec.get('label'))
        if y not in (0,1):
            raise ValueError(f"Unexpected label {y}; expected 0/1.")
        eeg58 = eeg_T[idx_58, :]

        segs, hop_sec = segment_continuous(eeg58, sfreq, window_sec, overlap_sec)
        if segs.size == 0 and keep_short_as_full:
            segs = np.expand_dims(eeg58.T, axis=0)  # full trial as one seg

        for s in segs:
            plv = compute_plv(s)  # [C,C]
            data = plv_to_graph(plv, TOPK_PERCENT, EPSILON)
            data.y = torch.tensor([y], dtype=torch.long)
            data.t = int(t_counter)
            t_counter += 1
            graphs.append(data)

    if not graphs:
        raise RuntimeError("No graphs built with the current window/overlap settings.")
    return graphs

# =============================
# Eval / Fine-tune pipeline
# =============================
def evaluate(model, loader, device):
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
        if cnt == 0:
            weights.append(0.0)  # ignore missing class
        else:
            weights.append(N / (num_classes * cnt))
    return torch.tensor(weights, dtype=torch.float32, device=device)

def run_pipeline(trials, sfreq, sd_path, out_dir, window_sec, overlap_sec, keep_short, finetune_frac, epochs, batch_size, lr, wd, device):
    graphs = build_graphs(trials, sfreq, window_sec, overlap_sec, keep_short)

    # Diagnostics on built graphs
    y_all = np.array([int(g.y.item()) for g in graphs], dtype=int)
    counts = dict(Counter(y_all))
    eff_hop_sec = max(1.0/sfreq, window_sec - overlap_sec)
    print(f"\nSegmentation → window={window_sec:.3f}s, overlap={overlap_sec:.3f}s, hop≈{eff_hop_sec:.3f}s")
    print(f"Built {len(graphs)} graphs. Class counts: {counts}")

    # Foundation test over ALL graphs
    full_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    sd = torch.load(sd_path, map_location=device)
    in_ch, h1, h2, h3, heads = infer_dims(sd)
    model = SimpleGAT(in_ch, h1, h2, h3, heads).to(device)
    model.load_state_dict(sd)
    foundation_acc, foundation_cm = evaluate(model, full_loader, device)
    print(f"Foundation — Acc: {foundation_acc:.2%}, CM=\n{foundation_cm}")

    # Temporal per-class split (no balancing; rely on class weights)
    tr_idx, te_idx = temporal_split_first_frac_per_class(graphs, y_all, finetune_frac)
    train_graphs = [graphs[i] for i in tr_idx]
    test_graphs  = [graphs[i] for i in te_idx]

    train_counts = Counter([int(g.y.item()) for g in train_graphs])
    test_counts  = Counter([int(g.y.item()) for g in test_graphs])
    print(f"Train class counts: {dict(train_counts)} | Test class counts: {dict(test_counts)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    model_ft = SimpleGAT(in_ch, h1, h2, h3, heads).to(device)
    model_ft.load_state_dict(sd)
    model_ft.train()

    # Loss: use weights only if both classes present
    if len(train_counts.keys()) < 2:
        print("⚠️ Train split single-class after repair; using unweighted CE.")
        crit = nn.CrossEntropyLoss()
    else:
        class_w = class_weight_tensor(train_graphs, device, num_classes=2)
        if class_w.numel() != 2 or (class_w == 0).sum().item() > 0:
            print(f"⚠️ Using unweighted CE (class_w={class_w.tolist()}).")
            crit = nn.CrossEntropyLoss()
        else:
            crit = nn.CrossEntropyLoss(weight=class_w)

    opt  = torch.optim.Adam(model_ft.parameters(), lr=lr, weight_decay=wd)

    best_acc = -1.0
    best_sd  = None
    best_cm  = None
    for ep in range(1, epochs+1):
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
        tr_acc = correct / max(1,total)
        te_acc, te_cm = evaluate(model_ft, test_loader, device)
        print(f"Epoch {ep}/{epochs} — Train {tr_acc:.2%} | Test {te_acc:.2%}")
        if te_acc > best_acc:
            best_acc = te_acc
            best_sd = model_ft.state_dict()
            best_cm = te_cm

    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, f"gat_finetuned_win{window_sec:g}_ov{overlap_sec:.2f}s.pt")
    if best_sd is not None:
        torch.save(best_sd, pt_path)
        print(f"✅ Saved best fine-tuned model → {pt_path} (Test acc {best_acc:.2%})")

    return {
        'window_sec': float(window_sec),
        'overlap_sec': float(overlap_sec),
        'keep_short_as_full': bool(keep_short),
        'foundation_acc': float(foundation_acc),
        'foundation_cm': foundation_cm.tolist(),
        'finetune_frac': float(finetune_frac),
        'train_class_counts': dict(train_counts),
        'test_class_counts': dict(test_counts),
        'finetune_best_acc': float(best_acc),
        'finetune_best_cm': best_cm.tolist() if best_cm is not None else None,
        'n_graphs': int(len(graphs)),
        'n_train_graphs': int(len(train_graphs)),
        'n_test_graphs': int(len(test_graphs)),
        'model_saved_to': pt_path if best_sd is not None else None
    }

# =============================
# Main
# =============================
def main():
    # Validate paths
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

    results = run_pipeline(
        trials=trials,
        sfreq=sfreq,
        sd_path=FOUNDATION_PT,
        out_dir=OUT_DIR,
        window_sec=WINDOW_SEC,
        overlap_sec=OVERLAP_SEC,
        keep_short=KEEP_SHORT_AS_FULL,
        finetune_frac=FINETUNE_FRAC,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        wd=WEIGHT_DECAY,
        device=device,
    )

    report = {
        'session_pkl': SESSION_PKL,
        'foundation_pt': FOUNDATION_PT,
        'sfreq': sfreq,
        'config': {
            'window_sec': WINDOW_SEC,
            'overlap_sec': OVERLAP_SEC,
            'keep_short_as_full': KEEP_SHORT_AS_FULL,
            'finetune_frac': FINETUNE_FRAC,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'topk_percent': TOPK_PERCENT,
            'epsilon': EPSILON,
        },
        'results': results,
    }
    report_path = os.path.join(OUT_DIR, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📝 Wrote report → {report_path}")

if __name__ == '__main__':
    main()
