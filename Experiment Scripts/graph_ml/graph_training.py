# graph_training.py
import os
import json
import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm
from scipy.signal import hilbert

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import (
    SUBJECT_ID, TRAIN_SESSION_ID, CURRENT_SESSION_ID,
    TRAIN_SESSION_PKL, MODEL_OUT_DIR,
    SAMPLING_RATE, WINDOW_SIZE, STEP_SIZE,
    EPSILON, KL_NBINS, KL_EPS, CV_EPS,
    CV_KEEP_PCTL, KL_KEEP_PCTL,
)

from preprocess import HEADSET_64, SHARED_58, idx_map
from preprocess import preprocess_window  # uses your bandpass/zscore settings

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

def segment_trial_to_windows(eeg_t64: np.ndarray, win: int, hop: int):
    """
    eeg_t64: [T,64] (we'll transpose from saved [64,T])
    returns: list of [win,64]
    """
    T = eeg_t64.shape[0]
    if T < win:
        return []
    out = []
    for start in range(0, T - win + 1, hop):
        out.append(eeg_t64[start:start+win].astype(np.float32))
    return out

def build_windows_from_trials(trials):
    """
    Builds windows in the SAME cadence as online (WINDOW_SIZE, STEP_SIZE).
    Returns:
      X_win: [N, win, 64]
      y:     [N]
      t:     [N] temporal index for per-class temporal split
    """
    win = int(WINDOW_SIZE)
    hop = max(1, int(STEP_SIZE))

    X_list, y_list, t_list = [], [], []
    t_counter = 0

    for rec in tqdm(trials, desc="Building windows (graph ML)"):
        eeg_ct = rec.get('eeg', None)  # [64, T]
        if eeg_ct is None or eeg_ct.shape[0] != len(HEADSET_64):
            continue

        y = int(rec.get('label'))
        if y not in (0, 1):
            continue

        eeg_t64 = eeg_ct.T.astype(np.float32)  # [T,64]

        wins = segment_trial_to_windows(eeg_t64, win=win, hop=hop)
        for w in wins:
            X_list.append(w)
            y_list.append(y)
            t_list.append(t_counter)
            t_counter += 1

    if not X_list:
        raise RuntimeError("No windows produced. Check WINDOW_SIZE vs trial length.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=int)
    t = np.asarray(t_list, dtype=int)
    return X, y, t

def temporal_split_per_class_train_val_test(t_all, y_all, train_frac=0.40, val_frac=0.30):
    y_all = np.asarray(y_all).ravel()
    t_all = np.asarray(t_all).ravel()
    classes = sorted(set(int(v) for v in y_all.tolist()))
    by_class = {}

    for c in classes:
        idx_c = np.where(y_all == c)[0]
        order = np.argsort(t_all[idx_c], kind='stable')
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

    train_idx = train_idx[np.argsort(t_all[train_idx])]
    val_idx   = val_idx[np.argsort(t_all[val_idx])]
    test_idx  = test_idx[np.argsort(t_all[test_idx])]
    return train_idx, val_idx, test_idx

# ─────────────────────────────────────────────────────────────────────────
# PLV (vectorized) + transform + feature extraction
# ─────────────────────────────────────────────────────────────────────────
def compute_plv_matrix(window_t58: np.ndarray):
    """
    window_t58: [T,58]
    Fast PLV:
      u = exp(1j*phase) => plv = abs((u^H u)/T)
    """
    analytic = hilbert(window_t58, axis=0)
    phase = np.angle(analytic)
    u = np.exp(1j * phase)  # [T,C]
    T = u.shape[0]
    plv = np.abs((u.conj().T @ u) / max(1, T)).astype(np.float32)  # [C,C]
    np.fill_diagonal(plv, 1.0)
    return plv

def plv_transform(plv_raw: np.ndarray, eps: float):
    plv = plv_raw.astype(np.float32).copy()
    np.fill_diagonal(plv, 0.0)
    X = -np.log(1.0 - plv + eps).astype(np.float32)
    np.fill_diagonal(X, 0.0)
    return X

def plv_features_from_window(window_t64: np.ndarray, eps: float):
    """
    window_t64: [T,64]
    -> preprocess to [T,58]
    -> PLV
    -> transformed matrix X
    -> edge feats (upper tri) + node feats (strength)
    """
    w58 = preprocess_window(window_t64)
    if w58 is None:
        return None, None

    plv = compute_plv_matrix(w58)            # [58,58]
    X = plv_transform(plv, eps=eps)          # [58,58]

    triu = np.triu_indices(X.shape[0], k=1)
    edges = X[triu].astype(np.float32)       # [E]
    nodes = np.sum(np.abs(X), axis=1).astype(np.float32)  # [58]
    return edges, nodes

# ─────────────────────────────────────────────────────────────────────────
# Stability / discriminability metrics
# ─────────────────────────────────────────────────────────────────────────
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

def select_features_by_cv_kl(cv, kl, cv_keep_pctl=30, kl_keep_pctl=70):
    cv_thr = float(np.percentile(cv, cv_keep_pctl))
    kl_thr = float(np.percentile(kl, kl_keep_pctl))
    sel = np.where((cv <= cv_thr) & (kl >= kl_thr))[0]
    return sel.astype(int), cv_thr, kl_thr

def apply_feature_mask(F_edges, F_nodes, sel_e, sel_n):
    Fe = F_edges[:, sel_e] if sel_e.size > 0 else np.zeros((F_edges.shape[0], 0), dtype=np.float32)
    Fn = F_nodes[:, sel_n] if sel_n.size > 0 else np.zeros((F_nodes.shape[0], 0), dtype=np.float32)
    return np.concatenate([Fe, Fn], axis=1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────
# Classical model zoo
# ─────────────────────────────────────────────────────────────────────────
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

def fit_grid_on_train_select_by_val(Xtr, ytr, Xva, yva, Xte, yte):
    best = {"name": None, "val_acc": -1.0, "test_acc": None, "best_params": None, "estimator": None}
    all_results = []

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
            "best_params": gs.best_params_,
        })

        if va_acc > best["val_acc"]:
            best.update({
                "name": name,
                "val_acc": float(va_acc),
                "test_acc": float(te_acc),
                "best_params": gs.best_params_,
                "estimator": gs.best_estimator_
            })

    return best, all_results

def train_and_save():
    trials = load_trials_one_session(TRAIN_SESSION_PKL)
    Xwin, y, t = build_windows_from_trials(trials)
    print(f"[GRAPH TRAIN] windows={len(Xwin)} class_counts={dict(Counter(y))}")

    tr_idx, va_idx, te_idx = temporal_split_per_class_train_val_test(t, y, train_frac=0.40, val_frac=0.30)

    # ── Extract PLV features per window
    edges_list, nodes_list = [], []
    for w in tqdm(Xwin, desc="Computing PLV features per window"):
        e, n = plv_features_from_window(w, eps=EPSILON)
        if e is None:
            raise RuntimeError("preprocess_window returned None unexpectedly.")
        edges_list.append(e)
        nodes_list.append(n)

    F_edges = np.stack(edges_list, axis=0).astype(np.float32)  # [N, E]
    F_nodes = np.stack(nodes_list, axis=0).astype(np.float32)  # [N, 58]

    # Split
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
    Fe_tr, Fe_va, Fe_te = F_edges[tr_idx], F_edges[va_idx], F_edges[te_idx]
    Fn_tr, Fn_va, Fn_te = F_nodes[tr_idx], F_nodes[va_idx], F_nodes[te_idx]

    # ── TRAIN-only CV/KL
    cv_e, kl_e, cv_n, kl_n = compute_cv_kl_landscape(Fe_tr, Fn_tr, ytr)

    sel_e, cv_thr_e, kl_thr_e = select_features_by_cv_kl(cv_e, kl_e, CV_KEEP_PCTL, KL_KEEP_PCTL)
    sel_n, cv_thr_n, kl_thr_n = select_features_by_cv_kl(cv_n, kl_n, CV_KEEP_PCTL, KL_KEEP_PCTL)

    print(f"[GRAPH TRAIN] Edge feats: total={F_edges.shape[1]}, selected={sel_e.size}")
    print(f"[GRAPH TRAIN] Node feats: total={F_nodes.shape[1]}, selected={sel_n.size}")
    if sel_e.size + sel_n.size == 0:
        raise RuntimeError("Selected 0 features. Loosen CV_KEEP_PCTL / KL_KEEP_PCTL.")

    # Apply selection
    Xtr = apply_feature_mask(Fe_tr, Fn_tr, sel_e, sel_n)
    Xva = apply_feature_mask(Fe_va, Fn_va, sel_e, sel_n)
    Xte = apply_feature_mask(Fe_te, Fn_te, sel_e, sel_n)

    best, all_results = fit_grid_on_train_select_by_val(Xtr, ytr, Xva, yva, Xte, yte)
    print(f"[GRAPH TRAIN] BEST={best['name']} val={best['val_acc']:.3f} test={best['test_acc']:.3f}")

    # Save selection stats (npz) and model pack (pkl)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    fs_path = os.path.join(MODEL_OUT_DIR, "selected_features_CVKL.npz")
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

    pack = {
        "clf": best["estimator"],  # sklearn Pipeline(scaler+clf)
        "sel_edge_idx": sel_e,
        "sel_node_idx": sel_n,
        "meta": {
            "subject_id": SUBJECT_ID,
            "train_session_id": TRAIN_SESSION_ID,
            "current_session_id": CURRENT_SESSION_ID,
            "train_session_pkl": TRAIN_SESSION_PKL,
            "sampling_rate": int(SAMPLING_RATE),
            "window_size_samples": int(WINDOW_SIZE),
            "step_size_samples": int(STEP_SIZE),
            "eps": float(EPSILON),
            "cv_keep_pctl": int(CV_KEEP_PCTL),
            "kl_keep_pctl": int(KL_KEEP_PCTL),
            "cv_thr_edges": float(cv_thr_e),
            "kl_thr_edges": float(kl_thr_e),
            "cv_thr_nodes": float(cv_thr_n),
            "kl_thr_nodes": float(kl_thr_n),
            "best_model": best["name"],
            "best_params": best["best_params"],
            "selection_npz": fs_path,
            "n_selected": int(sel_e.size + sel_n.size),
        },
        "all_results": all_results,
    }

    model_path = os.path.join(MODEL_OUT_DIR, "graph_ml_best.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pack, f)

    report_path = os.path.join(MODEL_OUT_DIR, "graph_ml_training_report.json")
    with open(report_path, "w") as f:
        json.dump({"best": pack["meta"], "all_results": all_results}, f, indent=2)

    return model_path, pack

if __name__ == "__main__":
    train_and_save()
