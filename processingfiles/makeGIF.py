# README
# EEG PLV Pipeline
#
# This script computes second-by-second Phase-Locking Value (PLV) maps from EEG data stored in
# `.mat` and `.pkl` formats, then generates both static PNG heatmaps and animated GIFs for each recording.
#
# **Data Input Directory:**
#   C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts
#
# **Processing Script Directory (Current Working Dir):**
#   C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\processingfiles
#
# Dataset Structure:
# processingfiles/ (current dir)
# └── processingData.py        # this script
#
# working scripts/ (input data)
# ├── neurofeedback/nf_results/Subject_000/Session_001/
# │   ├─ nf_1.mat                 # MATLAB v7.3 or v7 file with variable `y` (samples×channels)
# │   └─ manual_game_data.pkl     # list of dicts with `eeg_wins` windows (256×channels)
# ├── training/
# │   ├─ training_1.mat           # Raw EEG as .mat
# │   └─ training_data.pkl        # 3D array of EEG (epochs×times×channels)
# ├── online/collection_results/Subject_000/Session_001/
# │   ├─ Subject00S1.mat          # Raw EEG .mat
# │   └─ session_data.pkl         # Pickle list with `eeg_wins` or raw array
# └── online/PoC/Subject_000/Session_001/
#     ├─ Testing_Map_Stability.mat
#     └─ session_data.pkl         # PoC output, same structure
#
# Usage:
#   cd processingfiles
#   python processingData.py
#
# Outputs saved under `plv_outputs/` inside processingfiles:
# plv_outputs/
# ├── nf/mat/nf_1/
# │   ├─ plv_nf_mat_nf_1_win001.png
# │   └─ nf_mat_nf_1.gif
# └── nf/pkl/manual_game_data/
#     └─ manual_game_data.gif
# (similar for training, online_real, online_poc)

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # use v2 API
from scipy.signal import hilbert
from scipy.io import loadmat
import h5py

# Sampling frequency (Hz)
FS = 256
WINDOW_SAMPLES = FS

# Paths
data_dir = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts"
output_root = os.getcwd()  # processingfiles folder


def load_data(file_path):
    """
    Load EEG data from .mat or .pkl.
    Returns array of shape (channels, samples).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.mat':
        try:
            mat = loadmat(file_path)
            arr = mat.get('y') if 'y' in mat else next(v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2)
        except NotImplementedError:
            with h5py.File(file_path, 'r') as f:
                arr = f['y'][:] if 'y' in f else next(ds[:] for ds in f.values() if isinstance(ds, h5py.Dataset) and ds.ndim == 2)
        data = arr.T
    elif ext == '.pkl':
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, np.ndarray) and loaded.ndim == 3:
            ep, tm, ch = loaded.shape
            data = loaded.transpose(2, 0, 1).reshape(ch, ep * tm)
        elif isinstance(loaded, list) and loaded and isinstance(loaded[0], dict) and 'eeg_wins' in loaded[0]:
            windows = []
            for trial in loaded:
                for win in trial['eeg_wins']:
                    arr = np.array(win)
                    windows.append(arr.T)
            data = np.hstack(windows)
        elif isinstance(loaded, np.ndarray) and loaded.ndim == 2:
            data = loaded.T
        elif isinstance(loaded, dict) and 'data' in loaded:
            arr = np.array(loaded['data'])
            if arr.ndim == 2:
                data = arr.T
            elif arr.ndim == 3:
                ep, tm, ch = arr.shape
                data = arr.transpose(2, 0, 1).reshape(ch, ep * tm)
            else:
                raise ValueError(f"Unsupported data shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported pickle format: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    if data.ndim != 2:
        raise ValueError(f"Loaded data must be 2D; got {data.shape}")
    return data


def compute_plv(seg):
    analytic = hilbert(seg, axis=1)
    phases = np.angle(analytic)
    n_ch = phases.shape[0]
    plv = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i, n_ch):
            diff = phases[i] - phases[j]
            val = np.abs(np.mean(np.exp(1j * diff)))
            plv[i, j] = plv[j, i] = val
    return plv


def make_gif(frames_dir, gif_path, fps=1):
    pngs = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if not pngs:
        print(f"No frames in {frames_dir}, skipping GIF.")
        return
    with imageio.get_writer(gif_path, mode='I', duration=1/fps) as writer:
        for fn in pngs:
            writer.append_data(imageio.imread(fn))
    print(f"Saved GIF: {gif_path}")


def process_file(file_path, out_dir, label):
    eeg = load_data(file_path)
    n_win = eeg.shape[1] // WINDOW_SAMPLES
    os.makedirs(out_dir, exist_ok=True)
    for w in range(n_win):
        seg = eeg[:, w*WINDOW_SAMPLES:(w+1)*WINDOW_SAMPLES]
        plv = compute_plv(seg)
        plt.imshow(plv, vmin=0, vmax=1)
        plt.title(f"{label} Win {w+1}/{n_win}")
        plt.colorbar(label='PLV')
        plt.savefig(os.path.join(out_dir, f"plv_{label}_win{w+1:03d}.png"))
        plt.close()
    make_gif(frames_dir=out_dir, gif_path=os.path.join(out_dir, f"{label}.gif"))


def main():
    conditions = {
        'nf': os.path.join(data_dir, 'neurofeedback', 'nf_results', 'Subject_000', 'Session_001'),
        'training': os.path.join(data_dir, 'training'),
        'online_real': os.path.join(data_dir, 'online', 'collection_results', 'Subject_000', 'Session_001'),
        'online_poc': os.path.join(data_dir, 'online', 'PoC', 'Subject_000', 'Session_001')
    }
    for cond, path in conditions.items():
        for ext in ['mat', 'pkl']:
            for fp in glob.glob(os.path.join(path, f"*.{ext}")):
                name = os.path.splitext(os.path.basename(fp))[0]
                outd = os.path.join(output_root, 'plv_outputs', cond, ext, name)
                label = f"{cond}_{ext}_{name}"
                try:
                    process_file(fp, outd, label)
                except Exception as e:
                    print(f"Error processing {fp}: {e}")
    print(f"Done: PLV maps & GIFs saved under {os.path.join(output_root, 'plv_outputs')}.")

if __name__ == '__main__':
    main()

#%%
