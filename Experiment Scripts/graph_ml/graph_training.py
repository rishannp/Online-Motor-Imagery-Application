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

from config import (
    SUBJECT_ID, TRAIN_SESSION_ID, CURRENT_SESSION_ID,
    TRAIN_SESSION_PKLS, MODEL_OUT_DIR,
    SAMPLING_RATE, WINDOW_SIZE, STEP_SIZE,
    APPLY_BANDPASS,
)


try:
    from config import (
        EPSILON, KL_NBINS, KL_EPS, CV_EPS,
        CV_KEEP_PCTL, KL_KEEP_PCTL,
    )
except Exception:
    EPSILON = 1e-6
    KL_NBINS = 64
    KL_EPS = 1e-12
    CV_EPS = 1e-12
    CV_KEEP_PCTL = 20
    KL_KEEP_PCTL = 20
    print(
        "[GRAPH TRAIN][WARN] Missing graph params in config.py. Using defaults.",
        flush=True
    )

from preprocess import (
    HEADSET_64,
    preprocess_trial_58,
)

# ─────────────────────────────────────────────────────────────────────────
# Helpers: windowing, PLV features, stats, selection
# ─────────────────────────────────────────────────────────────────────────

def segment_trial_to_windows(eeg_tc: np.ndarray, win: int, hop: int):
    """eeg_tc: [T, C] -> windows: [N, win, C]"""
    T, C = eeg_tc.shape
    if T < win:
        return np.empty((0, win, C), dtype=np.float32)
    out = []
    for start in range(0, T - win + 1, hop):
        out.append(eeg_tc[start:start + win].astype(np.float32))
    return np.stack(out, axis=0)


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


def plv_features_from_window_58(window_t58: np.ndarray, eps: float):
    if window_t58 is None or window_t58.ndim != 2 or window_t58.shape[1] != 58:
        return None, None
    plv = compute_plv_matrix(window_t58)
    X = plv_transform(plv, eps=eps)
    triu = np.triu_indices(X.shape[0], k=1)
    edges = X[triu].astype(np.float32)
    nodes = np.sum(np.abs(X), axis=1).astype(np.float32)
    return edges, nodes


# ─────────────────────────────────────────────────────────────────────────
# Stability metrics (CV + symmetric KL)
# ─────────────────────────────────────────────────────────────────────────

def compute_cv(vec, eps=CV_EPS):
    mu = float(np.mean(vec))
    sd = float(np.std(vec))
    denom = abs(mu) + float(eps)
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
    ha, _ = np.histogram(a, bins=int(nbins), range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=int(nbins), range=(lo, hi), density=False)
    ha = ha.astype(np.float64) + float(eps)
    hb = hb.astype(np.float64) + float(eps)
    pa = ha / np.sum(ha)
    pb = hb / np.sum(hb)
    kl_ab = np.sum(pa * np.log(pa / pb))
    kl_ba = np.sum(pb * np.log(pb / pa))
    return float(0.5 * (kl_ab + kl_ba))


def compute_cv_kl_landscape(Fe_tr: np.ndarray, Fn_tr: np.ndarray, ytr: np.ndarray):
    ytr = np.asarray(ytr).astype(int)
    idx0 = np.where(ytr == 0)[0]
    idx1 = np.where(ytr == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        raise RuntimeError("TRAIN split missing a class; cannot compute KL.")
    E = Fe_tr.shape[1]
    cv_e = np.zeros((E,), dtype=np.float32)
    kl_e = np.zeros((E,), dtype=np.float32)
    for j in range(E):
        v = Fe_tr[:, j]
        cv_e[j] = compute_cv(v)
        kl_e[j] = sym_kl_hist(v[idx0], v[idx1])
    C = Fn_tr.shape[1]
    cv_n = np.zeros((C,), dtype=np.float32)
    kl_n = np.zeros((C,), dtype=np.float32)
    for j in range(C):
        v = Fn_tr[:, j]
        cv_n[j] = compute_cv(v)
        kl_n[j] = sym_kl_hist(v[idx0], v[idx1])
    return cv_e, kl_e, cv_n, kl_n


def select_features_by_cv_kl(cv: np.ndarray, kl: np.ndarray, cv_keep_pctl: int, kl_keep_pctl: int):
    cv = np.asarray(cv).astype(np.float32)
    kl = np.asarray(kl).astype(np.float32)
    cv_thr = float(np.percentile(cv, float(cv_keep_pctl)))
    kl_thr = float(np.percentile(kl, 100.0 - float(kl_keep_pctl)))
    keep = np.where((cv <= cv_thr) & (kl >= kl_thr))[0].astype(int)
    return keep, cv_thr, kl_thr


def apply_feature_mask(Fe: np.ndarray, Fn: np.ndarray, sel_e: np.ndarray, sel_n: np.ndarray):
    Fe2 = Fe[:, sel_e] if sel_e.size > 0 else np.zeros((Fe.shape[0], 0), dtype=np.float32)
    Fn2 = Fn[:, sel_n] if sel_n.size > 0 else np.zeros((Fn.shape[0], 0), dtype=np.float32)
    return np.concatenate([Fe2, Fn2], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Trial-level split
# ─────────────────────────────────────────────────────────────────────────

def _stratified_trial_split(trials, train_frac=0.40, val_frac=0.30, seed=0):
    """Split by trial index (not window) to prevent leakage from overlapping windows."""
    rng = np.random.default_rng(int(seed))
    idx0 = [i for i, r in enumerate(trials) if int(r.get("label", -1)) == 0]
    idx1 = [i for i, r in enumerate(trials) if int(r.get("label", -1)) == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def _split_one(idxs):
        n = len(idxs)
        n_tr = int(np.ceil(train_frac * n))
        n_va = int(np.ceil(val_frac * n))
        return idxs[:n_tr], idxs[n_tr:n_tr + n_va], idxs[n_tr + n_va:]

    tr0, va0, te0 = _split_one(idx0)
    tr1, va1, te1 = _split_one(idx1)
    tr, va, te = tr0 + tr1, va0 + va1, te0 + te1
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


# ─────────────────────────────────────────────────────────────────────────
# Session loading
# ─────────────────────────────────────────────────────────────────────────

def load_trials_one_session(session_pkl_path):
    if not os.path.isfile(session_pkl_path):
        raise FileNotFoundError(f"Missing session file: {session_pkl_path}")
    with open(session_pkl_path, "rb") as f:
        d = pickle.load(f)
    trials = []
    for _, rec in d.items():
        eeg = rec.get("eeg", None)
        if eeg is None or getattr(eeg, "size", 0) == 0:
            continue
        trials.append(rec)
    if not trials:
        raise RuntimeError(f"No usable trials in: {session_pkl_path}")
    return trials


def load_trials_all_sessions(train_session_pkls):
    """Pool trials from all resolved sessions, preserving load order (chronological within each app)."""
    all_trials = []
    for app_name, session_id, pkl_path in train_session_pkls:
        print(f"[GRAPH TRAIN] Loading {app_name} session {session_id}: {pkl_path}", flush=True)
        session_trials = load_trials_one_session(pkl_path)
        print(f"[GRAPH TRAIN]   -> {len(session_trials)} usable trials", flush=True)
        all_trials.extend(session_trials)
    if not all_trials:
        raise RuntimeError("No usable trials found across all configured sessions.")
    print(f"[GRAPH TRAIN] Total trials pooled: {len(all_trials)}", flush=True)
    return all_trials


# ─────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────

def build_search_space():
    models = []
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
    models.append(("LogReg", lr, lr_grid))

    svm = Pipeline([("scaler", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
    svm_grid = {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"], "clf__gamma": ["scale"]}
    models.append(("SVC", svm, svm_grid))

    knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    knn_grid = {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]}
    models.append(("KNN", knn, knn_grid))

    return models


def fit_grid_on_train_select_by_val(Xtr, ytr, Xva, yva, Xte, yte):
    models = build_search_space()
    all_results = []
    best = None
    for name, est, grid in models:
        gs = GridSearchCV(est, param_grid=grid, scoring="accuracy", cv=3, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        va_acc = float(accuracy_score(yva, gs.best_estimator_.predict(Xva)))
        te_acc = float(accuracy_score(yte, gs.best_estimator_.predict(Xte)))
        all_results.append({"name": name, "best_params": gs.best_params_, "val_acc": va_acc, "test_acc": te_acc, "estimator": gs.best_estimator_})
        if best is None or va_acc > best["val_acc"]:
            best = all_results[-1]
        print(f"[GRAPH TRAIN] {name} best={gs.best_params_} val={va_acc:.3f} test={te_acc:.3f}", flush=True)
    return best, all_results


# ─────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────

def train_graph_ml():
    trials = load_trials_all_sessions(TRAIN_SESSION_PKLS)

    tr_trials_idx, va_trials_idx, te_trials_idx = _stratified_trial_split(trials, seed=0)
    print(f"[GRAPH TRAIN] trials split: train={len(tr_trials_idx)} val={len(va_trials_idx)} test={len(te_trials_idx)}", flush=True)

    win = int(WINDOW_SIZE)
    hop = max(1, int(STEP_SIZE))

    windows_58, y, trial_ids = [], [], []
    for t_idx, rec in enumerate(tqdm(trials, desc="Preprocess trials + segment windows")):
        eeg_ct = rec.get("eeg", None)
        if eeg_ct is None or eeg_ct.shape[0] != len(HEADSET_64):
            continue
        lab = int(rec.get("label", -1))
        if lab not in (0, 1):
            continue
        eeg_t64 = eeg_ct.T.astype(np.float32)
        eeg_t58 = preprocess_trial_58(eeg_t64)
        if eeg_t58 is None:
            continue
        wins = segment_trial_to_windows(eeg_t58, win=win, hop=hop)
        for w in wins:
            windows_58.append(w)
            y.append(lab)
            trial_ids.append(t_idx)

    if not windows_58:
        raise RuntimeError("No windows built. Check WINDOW_SIZE/STEP_SIZE and your input trials.")

    windows_58 = np.asarray(windows_58, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    trial_ids = np.asarray(trial_ids, dtype=int)
    print(f"[GRAPH TRAIN] windows={len(y)} label_counts={Counter(y)} hop={hop} win={win}", flush=True)

    tr_idx = np.where(np.isin(trial_ids, np.array(tr_trials_idx, dtype=int)))[0]
    va_idx = np.where(np.isin(trial_ids, np.array(va_trials_idx, dtype=int)))[0]
    te_idx = np.where(np.isin(trial_ids, np.array(te_trials_idx, dtype=int)))[0]

    if tr_idx.size == 0 or va_idx.size == 0 or te_idx.size == 0:
        raise RuntimeError("Empty split after windowing. Too few trials/windows.")

    edges_list, nodes_list = [], []
    for w in tqdm(windows_58, desc="Extract PLV features per window"):
        e, n = plv_features_from_window_58(w, eps=float(EPSILON))
        if e is None:
            raise RuntimeError("Feature extraction returned None unexpectedly.")
        edges_list.append(e)
        nodes_list.append(n)

    F_edges = np.stack(edges_list, axis=0).astype(np.float32)
    F_nodes = np.stack(nodes_list, axis=0).astype(np.float32)

    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
    Fe_tr, Fe_va, Fe_te = F_edges[tr_idx], F_edges[va_idx], F_edges[te_idx]
    Fn_tr, Fn_va, Fn_te = F_nodes[tr_idx], F_nodes[va_idx], F_nodes[te_idx]

    cv_e, kl_e, cv_n, kl_n = compute_cv_kl_landscape(Fe_tr, Fn_tr, ytr)
    sel_e, cv_thr_e, kl_thr_e = select_features_by_cv_kl(cv_e, kl_e, int(CV_KEEP_PCTL), int(KL_KEEP_PCTL))
    sel_n, cv_thr_n, kl_thr_n = select_features_by_cv_kl(cv_n, kl_n, int(CV_KEEP_PCTL), int(KL_KEEP_PCTL))

    print(f"[GRAPH TRAIN] Edge feats: total={F_edges.shape[1]}, selected={sel_e.size}", flush=True)
    print(f"[GRAPH TRAIN] Node feats: total={F_nodes.shape[1]}, selected={sel_n.size}", flush=True)
    if sel_e.size + sel_n.size == 0:
        raise RuntimeError("Feature selection picked 0 features. Relax CV_KEEP_PCTL/KL_KEEP_PCTL.")

    Xtr = apply_feature_mask(Fe_tr, Fn_tr, sel_e, sel_n)
    Xva = apply_feature_mask(Fe_va, Fn_va, sel_e, sel_n)
    Xte = apply_feature_mask(Fe_te, Fn_te, sel_e, sel_n)

    best, all_results = fit_grid_on_train_select_by_val(Xtr, ytr, Xva, yva, Xte, yte)
    print(f"[GRAPH TRAIN] BEST={best['name']} val={best['val_acc']:.3f} test={best['test_acc']:.3f}", flush=True)

    cm = confusion_matrix(yte, best["estimator"].predict(Xte))
    print("[GRAPH TRAIN] Test confusion matrix:\n", cm, flush=True)

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    np.savez(
        os.path.join(MODEL_OUT_DIR, "selected_features_CVKL.npz"),
        sel_edge_idx=sel_e, sel_node_idx=sel_n,
        cv_edges=cv_e, kl_edges=kl_e,
        cv_nodes=cv_n, kl_nodes=kl_n,
        cv_thr_edges=cv_thr_e, kl_thr_edges=kl_thr_e,
        cv_thr_nodes=cv_thr_n, kl_thr_nodes=kl_thr_n,
        cv_keep_pctl=int(CV_KEEP_PCTL), kl_keep_pctl=int(KL_KEEP_PCTL),
        train_trial_idx=np.array(tr_trials_idx, dtype=int),
        val_trial_idx=np.array(va_trials_idx, dtype=int),
        test_trial_idx=np.array(te_trials_idx, dtype=int),
    )

    pack = {
        "clf": best["estimator"],
        "sel_edge_idx": sel_e,
        "sel_node_idx": sel_n,
        "meta": {
            "subject_id": SUBJECT_ID,
            "train_sessions": [(a, s) for a, s, _ in TRAIN_SESSION_PKLS],
            "train_session_id": TRAIN_SESSION_ID,
            "current_session_id": CURRENT_SESSION_ID,
            "sampling_rate": float(SAMPLING_RATE),
            "window_size": int(WINDOW_SIZE),
            "step_size": int(STEP_SIZE),
            "apply_bandpass": bool(APPLY_BANDPASS),
            "epsilon": float(EPSILON),
            "cv_keep_pctl": int(CV_KEEP_PCTL),
            "kl_keep_pctl": int(KL_KEEP_PCTL),
            "best_model": best["name"],
            "best_params": best["best_params"],
            "val_acc": float(best["val_acc"]),
            "test_acc": float(best["test_acc"]),
        }
    }

    model_path = os.path.join(MODEL_OUT_DIR, "graph_ml_model_pack.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pack, f)

    with open(os.path.join(MODEL_OUT_DIR, "graph_training_summary.json"), "w") as f:
        json.dump(pack["meta"], f, indent=2)

    return model_path, pack


def train_and_save():
    return train_graph_ml()


if __name__ == "__main__":
    train_and_save()