# processing_pipeline.py
#
# Unified EEG processing pipeline for neurofeedback, training, and online sessions:
# 1) Second-by-second PLV maps (PNGs + GIFs)
# 2) Coefficient of Variation (CV) heatmaps across time (Red = Low CV, Green = High CV)
# 3) Grand-average ERS/ERD for Left vs. Right Motor Imagery with per-channel time series

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.signal import hilbert, butter, filtfilt
from scipy.io import loadmat
import h5py

# === Parameters ===
FS = 256               # Sampling frequency (Hz)
WINDOW_SAMPLES = FS    # 1-second windows
MU_BAND = (8, 12)      # Mu band for ERS/ERD
BASELINE_S = 1         # seconds used for baseline normalization

# === Paths ===
DATA_DIR = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts"
OUTPUT_ROOT = os.getcwd()  # processingfiles
PLV_DIR = os.path.join(OUTPUT_ROOT, 'plv_outputs')
CV_DIR  = os.path.join(OUTPUT_ROOT, 'cv_outputs')
ERS_DIR = os.path.join(OUTPUT_ROOT, 'ers_outputs')

# === Helpers ===
def get_subject_sessions():
    conds = ['neurofeedback/nf_results', 'training',
             'online/collection_results', 'online/PoC']
    sessions = {}
    for cond in conds:
        base = os.path.join(DATA_DIR, cond)
        for subj in glob.glob(os.path.join(base, 'Subject_*')):
            for sess in glob.glob(os.path.join(subj, 'Session_*')):
                key = cond.replace('/', '_')
                sessions.setdefault(key, []).append(sess)
    return sessions


def load_data(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext == '.mat':
        try:
            mat = loadmat(fp)
            arr = mat.get('y') or next(v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2)
        except NotImplementedError:
            with h5py.File(fp, 'r') as f:
                arr = f['y'][:] if 'y' in f else next(ds[:] for ds in f.values() if isinstance(ds, h5py.Dataset) and ds.ndim == 2)
        return arr.T
    with open(fp, 'rb') as f:
        ld = pickle.load(f)
    if isinstance(ld, list) and ld and isinstance(ld[0], dict) and 'eeg_wins' in ld[0]:
        segs = [np.array(win).T for trial in ld for win in trial['eeg_wins']]
        return np.hstack(segs)
    if isinstance(ld, np.ndarray) and ld.ndim == 3:
        e, t, ch = ld.shape
        return ld.transpose(2, 0, 1).reshape(ch, e * t)
    if isinstance(ld, np.ndarray) and ld.ndim == 2:
        return ld.T
    if isinstance(ld, dict) and 'data' in ld:
        arr = np.array(ld['data'])
        if arr.ndim == 2:
            return arr.T
        if arr.ndim == 3:
            e, t, ch = arr.shape
            return arr.transpose(2, 0, 1).reshape(ch, e * t)
    raise ValueError(f"Unsupported data format: {type(ld)}")


def compute_plv(seg):
    A = hilbert(seg, axis=1)
    ph = np.angle(A)
    n = ph.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            diff = ph[i] - ph[j]
            val = np.abs(np.mean(np.exp(1j * diff)))
            P[i, j] = P[j, i] = val
    return P


def bandpower_timecourse(seg):
    b, a = butter(4, [MU_BAND[0] / (FS / 2), MU_BAND[1] / (FS / 2)], btype='band')
    filt = filtfilt(b, a, seg, axis=1)
    env = np.abs(hilbert(filt, axis=1))
    return env  # channels x time

# === Pipeline Steps ===

def run_plv_cv():
    sessions = get_subject_sessions()
    for cond, sess_list in sessions.items():
        for sess in sess_list:
            for ext in ['mat', 'pkl']:
                for fp in glob.glob(os.path.join(sess, f"*.{ext}")):
                    name = os.path.splitext(os.path.basename(fp))[0]
                    data = load_data(fp)
                    nwin = data.shape[1] // WINDOW_SAMPLES
                    plv_stack = []
                    out_plv = os.path.join(PLV_DIR, cond, ext, name)
                    os.makedirs(out_plv, exist_ok=True)
                    for w in range(nwin):
                        seg = data[:, w * WINDOW_SAMPLES:(w + 1) * WINDOW_SAMPLES]
                        P = compute_plv(seg)
                        plv_stack.append(P)
                        plt.imshow(P, vmin=0, vmax=1, cmap='RdYlGn_r')
                        plt.title(f"{cond}_{name} W{w+1}/{nwin}")
                        plt.colorbar()
                        plt.savefig(os.path.join(out_plv, f"plv_{w+1:03d}.png")); plt.close()
                    with imageio.get_writer(os.path.join(out_plv, 'plv.gif'), mode='I', duration=1) as writer:
                        for img in sorted(glob.glob(os.path.join(out_plv, 'plv_*.png'))):
                            writer.append_data(imageio.imread(img))
                    stack = np.stack(plv_stack)
                    mu = stack.mean(0); sd = stack.std(0)
                    cv = np.zeros_like(mu); mask = mu > 0; cv[mask] = sd[mask] / mu[mask]
                    out_cv = os.path.join(CV_DIR, cond, ext)
                    os.makedirs(out_cv, exist_ok=True)
                    plt.imshow(cv, vmin=0, vmax=np.nanpercentile(cv, 95), cmap='RdYlGn')  # RED = low CV, GREEN = high CV
                    plt.title(f"CV_{cond}_{name}")
                    plt.colorbar()
                    plt.savefig(os.path.join(out_cv, f"cv_{name}.png")); plt.close()
                    print(f"PLV+CV done: {cond}/{ext}/{name}")

def run_ers_erd():
    sessions = {
        'training': os.path.join(DATA_DIR, 'training', 'training_data.pkl'),
        'online_real': os.path.join(DATA_DIR, 'online', 'collection_results', 'Subject_000', 'Session_001', 'session_data.pkl')
    }
    for label, path in sessions.items():
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if label == 'training':
            X = np.array(data['data'])  # trials x samples x channels
            Y = np.array(data['event_markers'])[:, 0].astype(int)
        else:
            X, Y = [], []
            for trial in data:
                eeg = np.stack(trial['eeg_wins'], axis=0).reshape(-1, 62)
                X.append(eeg)
                Y.append(trial['label'])
            X = np.array(X); Y = np.array(Y)
        L, R = [], []
        for x, y in zip(X, Y):
            seg = x.T
            bp = bandpower_timecourse(seg)
            (L if y == 0 else R).append(bp)
        L, R = np.stack(L), np.stack(R)
        baseline = np.mean(np.concatenate([L[:, :, :BASELINE_S*FS], R[:, :, :BASELINE_S*FS]], axis=0), axis=2, keepdims=True)
        ers = (L - baseline) / baseline
        erd = (R - baseline) / baseline
        meanL, stdL = ers.mean(0), ers.std(0)
        meanR, stdR = erd.mean(0), erd.std(0)
        out_dir = os.path.join(ERS_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        for ch in range(meanL.shape[0]):
            plt.figure()
            t = np.arange(meanL.shape[1]) / FS
            plt.plot(t, meanL[ch], label='Left', color='blue')
            plt.fill_between(t, meanL[ch] - stdL[ch], meanL[ch] + stdL[ch], color='blue', alpha=0.3)
            plt.plot(t, meanR[ch], label='Right', color='orange')
            plt.fill_between(t, meanR[ch] - stdR[ch], meanR[ch] + stdR[ch], color='orange', alpha=0.3)
            plt.axhline(0, linestyle='--', color='gray')
            plt.title(f"ERS/ERD - Channel {ch}")
            plt.xlabel("Time (s)"); plt.ylabel("ΔPower/Power")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"ch{ch:02d}.png")); plt.close()
        print(f"ERS/ERD done for {label}")

def main():
    os.makedirs(PLV_DIR, exist_ok=True)
    os.makedirs(CV_DIR, exist_ok=True)
    os.makedirs(ERS_DIR, exist_ok=True)
    run_plv_cv()
    run_ers_erd()
    print("All processing complete.")

if __name__ == '__main__':
    main()
