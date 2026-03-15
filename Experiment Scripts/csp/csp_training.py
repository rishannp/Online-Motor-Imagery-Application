# csp_training.py
import os
import json
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC

from config import (
    TRAIN_SESSION_PKLS, MODEL_OUT_DIR,
    SAMPLING_RATE,
    WINDOW_SIZE,
    CSP_NCOMP, CSP_REG,
    CSP_CHANNELS,
    SUBJECT_ID, TRAIN_SESSION_ID, CURRENT_SESSION_ID,
    APPLY_BANDPASS,
)

from preprocess import (
    HEADSET_64, SHARED_58, idx_map,
    bandpass_causal_trials_continuous,
)


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
        print(f"[CSP TRAIN] Loading {app_name} session {session_id}: {pkl_path}", flush=True)
        session_trials = load_trials_one_session(pkl_path)
        print(f"[CSP TRAIN]   -> {len(session_trials)} usable trials", flush=True)
        all_trials.extend(session_trials)
    if not all_trials:
        raise RuntimeError("No usable trials found across all configured sessions.")
    print(f"[CSP TRAIN] Total trials pooled: {len(all_trials)}", flush=True)
    return all_trials


def segment_trial_to_windows(eeg_ct: np.ndarray, win: int, hop: int):
    C, T = eeg_ct.shape
    if T < win:
        return np.empty((0, C, win), dtype=np.float32)
    out = []
    for start in range(0, T - win + 1, hop):
        out.append(eeg_ct[:, start:start + win].astype(np.float32))
    return np.stack(out, axis=0)


def _stratified_trial_split(trials, train_frac=0.40, val_frac=0.30, seed=0):
    rng = np.random.default_rng(int(seed))
    idx0 = [i for i, r in enumerate(trials) if int(r.get("label", -1)) == 0]
    idx1 = [i for i, r in enumerate(trials) if int(r.get("label", -1)) == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def _split_one(idxs):
        n = len(idxs)
        n_tr = int(np.ceil(train_frac * n)) if n else 0
        n_va = int(np.ceil(val_frac * n)) if n else 0
        tr = idxs[:n_tr]
        va = idxs[n_tr:n_tr + n_va]
        te = idxs[n_tr + n_va:]
        if len(tr) == 0 and len(va) > 0:
            tr.append(va.pop(0))
        if len(va) == 0 and len(te) > 0:
            va.append(te.pop(0))
        return tr, va, te

    tr0, va0, te0 = _split_one(idx0)
    tr1, va1, te1 = _split_one(idx1)
    tr, va, te = tr0 + tr1, va0 + va1, te0 + te1
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def build_windows_from_trials(trials, *, hop: int):
    idx_58  = np.array(idx_map(HEADSET_64, SHARED_58),   dtype=int)
    idx_csp = np.array(idx_map(SHARED_58,  CSP_CHANNELS), dtype=int)

    win = int(WINDOW_SIZE)
    hop = max(1, int(hop))

    eeg58_list, y_list_trial = [], []
    for rec in trials:
        eeg = rec.get("eeg", None)
        if eeg is None or eeg.shape[0] != len(HEADSET_64):
            continue
        y = int(rec.get("label"))
        if y not in (0, 1):
            continue
        eeg58_list.append(eeg[idx_58, :].astype(np.float32))
        y_list_trial.append(y)

    if not eeg58_list:
        raise RuntimeError("No usable trials to build CSP windows.")

    if APPLY_BANDPASS:
        eeg58_list = bandpass_causal_trials_continuous(eeg58_list)

    X_list, y_list, g_list = [], [], []
    for gi, (eeg58, y) in enumerate(tqdm(list(zip(eeg58_list, y_list_trial)), desc="Building CSP windows")):
        eegcsp = eeg58[idx_csp, :]
        wins = segment_trial_to_windows(eegcsp, win=win, hop=hop)
        for w in wins:
            X_list.append(w)
            y_list.append(y)
            g_list.append(gi)

    if not X_list:
        raise RuntimeError("No CSP windows produced. Check window size vs trial length.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=int)
    g = np.asarray(g_list, dtype=int)
    return X, y, g


def _cov_trial(X_CT, reg=CSP_REG):
    X = X_CT - X_CT.mean(axis=1, keepdims=True)
    cov = (X @ X.T) / max(1, X.shape[1] - 1)
    cov = cov / (np.trace(cov) + 1e-12)
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov


def fit_csp(X_train, y_train, n_components=CSP_NCOMP, reg=CSP_REG):
    from scipy.linalg import eigh
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    if len(X0) == 0 or len(X1) == 0:
        raise RuntimeError("CSP needs both classes in training set.")
    R0 = np.mean([_cov_trial(x, reg) for x in X0], axis=0)
    R1 = np.mean([_cov_trial(x, reg) for x in X1], axis=0)
    R  = R0 + R1
    R1 = 0.5 * (R1 + R1.T)
    R  = 0.5 * (R  + R.T)
    evals, evecs = eigh(R1, R)
    order = np.argsort(evals)[::-1]
    W = evecs[:, order].astype(np.float32)
    k  = int(n_components)
    k1 = k // 2
    k2 = k - k1
    picks = np.concatenate([np.arange(k2), np.arange(W.shape[1] - k1, W.shape[1])])
    return W, picks


def csp_transform(X, W, picks):
    feats = []
    for x in X:
        Z  = (W.T @ x)
        Zp = Z[picks, :]
        Zp = Zp - Zp.mean(axis=1, keepdims=True)
        var = np.var(Zp, axis=1)
        var = var / (np.sum(var) + 1e-12)
        feats.append(np.log(var + 1e-12))
    return np.stack(feats, axis=0).astype(np.float32)


def classical_model_zoo():
    zoo = {}
    zoo["svm_linear"] = (
        SVC(kernel="linear", class_weight="balanced", probability=False),
        {"clf__C": [0.1, 1.0, 10.0]},
    )
    return zoo


def fit_grid_on_train_select_by_val(Xtr, ytr, gtr, Xva, yva, Xte, yte):
    best = {"name": None, "val_acc": -1.0, "test_acc": None, "best_params": None, "estimator": None}
    all_results = []

    n_groups = int(np.unique(gtr).size)
    n_splits = min(5, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"Need >=2 trials/groups for GroupKFold, got {n_groups}.")

    cv = GroupKFold(n_splits=n_splits)

    for name, (base_clf, grid) in classical_model_zoo().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])
        gs = GridSearchCV(pipe, grid, scoring="accuracy", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)

        yhat_va = gs.best_estimator_.predict(Xva)
        va_acc  = accuracy_score(yva, yhat_va)
        va_cm   = confusion_matrix(yva, yhat_va, labels=[0, 1]).astype(int)

        yhat_te = gs.best_estimator_.predict(Xte)
        te_acc  = accuracy_score(yte, yhat_te)
        te_cm   = confusion_matrix(yte, yhat_te, labels=[0, 1]).astype(int)

        all_results.append({
            "model": name,
            "val_acc": float(va_acc), "val_cm": va_cm.tolist(),
            "test_acc": float(te_acc), "test_cm": te_cm.tolist(),
            "best_params": gs.best_params_,
        })

        if va_acc > best["val_acc"]:
            best.update({
                "name": name,
                "val_acc": float(va_acc), "test_acc": float(te_acc),
                "best_params": gs.best_params_,
                "estimator": gs.best_estimator_,
            })

    return best, all_results


def train_and_save():
    trials = load_trials_all_sessions(TRAIN_SESSION_PKLS)

    tr_trials, va_trials, te_trials = _stratified_trial_split(trials, train_frac=0.40, val_frac=0.30, seed=0)
    trials_tr = [trials[i] for i in tr_trials]
    trials_va = [trials[i] for i in va_trials]
    trials_te = [trials[i] for i in te_trials]

    hop = int(WINDOW_SIZE)
    Xtr, ytr, gtr = build_windows_from_trials(trials_tr, hop=hop)
    Xva, yva, _   = build_windows_from_trials(trials_va, hop=hop)
    Xte, yte, _   = build_windows_from_trials(trials_te, hop=hop)

    print(f"[CSP TRAIN] windows_tr={len(Xtr)} windows_va={len(Xva)} windows_te={len(Xte)}")
    print(
        f"[CSP TRAIN] train_class_counts={dict(Counter(ytr))} "
        f"val_class_counts={dict(Counter(yva))} "
        f"test_class_counts={dict(Counter(yte))}"
    )

    W, picks = fit_csp(Xtr, ytr, n_components=CSP_NCOMP, reg=CSP_REG)
    Ftr = csp_transform(Xtr, W, picks)
    Fva = csp_transform(Xva, W, picks)
    Fte = csp_transform(Xte, W, picks)

    best, all_results = fit_grid_on_train_select_by_val(Ftr, ytr, gtr, Fva, yva, Fte, yte)
    print(f"[CSP TRAIN] BEST={best['name']} val={best['val_acc']:.3f} test={best['test_acc']:.3f}")

    pack = {
        "W":     W,
        "picks": picks,
        "clf":   best["estimator"],
        "meta": {
            "subject_id":          SUBJECT_ID,
            "train_sessions":      [(a, s) for a, s, _ in TRAIN_SESSION_PKLS],
            "train_session_id":    TRAIN_SESSION_ID,
            "current_session_id":  CURRENT_SESSION_ID,
            "sampling_rate":       int(SAMPLING_RATE),
            "window_size_samples": int(WINDOW_SIZE),
            "train_hop_samples":   int(hop),
            "csp_channels":        list(CSP_CHANNELS),
            "csp_ncomp":           int(CSP_NCOMP),
            "csp_reg":             float(CSP_REG),
            "best_model":          best["name"],
            "best_params":         best["best_params"],
        },
        "trial_split": {"train": tr_trials, "val": va_trials, "test": te_trials},
        "all_results": all_results,
    }

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUT_DIR, "csp_best.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pack, f)

    with open(os.path.join(MODEL_OUT_DIR, "csp_training_report.json"), "w") as f:
        json.dump({"best": pack["meta"], "all_results": all_results}, f, indent=2)

    return model_path, pack