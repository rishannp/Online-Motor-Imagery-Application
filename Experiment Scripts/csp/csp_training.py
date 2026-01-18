# csp_training.py
import os
import json
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import (
    TRAIN_SESSION_PKL, MODEL_OUT_DIR,
    SAMPLING_RATE,
    WINDOW_SIZE, STEP_SIZE,
    CSP_NCOMP, CSP_REG,
    CSP_CHANNELS,
    SUBJECT_ID, TRAIN_SESSION_ID, CURRENT_SESSION_ID,
)

from preprocess import HEADSET_64, SHARED_58, idx_map

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

def segment_trial_to_windows(eeg_ct: np.ndarray, win: int, hop: int):
    C, T = eeg_ct.shape
    if T < win:
        return np.empty((0, C, win), dtype=np.float32)
    out = []
    for start in range(0, T - win + 1, hop):
        out.append(eeg_ct[:, start:start+win].astype(np.float32))
    return np.stack(out, axis=0)

def build_windows_from_trials(trials):
    idx_58  = np.array(idx_map(HEADSET_64, SHARED_58), dtype=int)
    idx_csp = np.array(idx_map(SHARED_58, CSP_CHANNELS), dtype=int)

    X_list, y_list, t_list = [], [], []
    t_counter = 0
    win = int(WINDOW_SIZE)
    hop = max(1, int(STEP_SIZE))

    for rec in tqdm(trials, desc="Building CSP windows"):
        eeg = rec.get('eeg', None)
        if eeg is None or eeg.shape[0] != len(HEADSET_64):
            continue

        y = int(rec.get('label'))
        if y not in (0, 1):
            continue

        eeg58  = eeg[idx_58, :]
        eegcsp = eeg58[idx_csp, :]

        wins = segment_trial_to_windows(eegcsp, win=win, hop=hop)
        for w in wins:
            X_list.append(w)
            y_list.append(y)
            t_list.append(t_counter)
            t_counter += 1

    if not X_list:
        raise RuntimeError("No CSP windows produced. Check window size vs trial length.")
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

    k = int(n_components)
    k1 = k // 2
    k2 = k - k1
    picks = np.concatenate([np.arange(k2), np.arange(W.shape[1] - k1, W.shape[1])])
    return W, picks

def csp_transform(X, W, picks):
    feats = []
    for x in X:
        Z = (W.T @ x)
        Zp = Z[picks, :]
        var = np.var(Zp, axis=1)
        var = var / (np.sum(var) + 1e-12)
        feats.append(np.log(var + 1e-12))
    return np.stack(feats, axis=0).astype(np.float32)

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
    X, y, t = build_windows_from_trials(trials)
    print(f"[CSP TRAIN] windows={len(X)} class_counts={dict(Counter(y))}")

    tr_idx, va_idx, te_idx = temporal_split_per_class_train_val_test(t, y, train_frac=0.40, val_frac=0.30)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    W, picks = fit_csp(Xtr, ytr, n_components=CSP_NCOMP, reg=CSP_REG)
    Ftr = csp_transform(Xtr, W, picks)
    Fva = csp_transform(Xva, W, picks)
    Fte = csp_transform(Xte, W, picks)

    best, all_results = fit_grid_on_train_select_by_val(Ftr, ytr, Fva, yva, Fte, yte)
    print(f"[CSP TRAIN] BEST={best['name']} val={best['val_acc']:.3f} test={best['test_acc']:.3f}")

    pack = {
        "W": W,
        "picks": picks,
        "clf": best["estimator"],
        "meta": {
            "subject_id": SUBJECT_ID,
            "train_session_id": TRAIN_SESSION_ID,
            "current_session_id": CURRENT_SESSION_ID,
            "train_session_pkl": TRAIN_SESSION_PKL,
            "sampling_rate": int(SAMPLING_RATE),
            "window_size_samples": int(WINDOW_SIZE),
            "step_size_samples": int(STEP_SIZE),
            "csp_channels": list(CSP_CHANNELS),
            "csp_ncomp": int(CSP_NCOMP),
            "csp_reg": float(CSP_REG),
            "best_model": best["name"],
            "best_params": best["best_params"],
        },
        "split_idx": {"train": tr_idx.tolist(), "val": va_idx.tolist(), "test": te_idx.tolist()},
        "all_results": all_results,
    }

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUT_DIR, "csp_best.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pack, f)

    report_path = os.path.join(MODEL_OUT_DIR, "csp_training_report.json")
    with open(report_path, "w") as f:
        json.dump({"best": pack["meta"], "all_results": all_results}, f, indent=2)

    return model_path, pack
