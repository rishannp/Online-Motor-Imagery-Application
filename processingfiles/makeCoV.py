# compute_plv_cvmap.py
#
# Compute Coefficient of Variation (CV) of PLV directly from EEG data windows
# Produces a 2D CV map (channels x channels) for each dataset without using PNG frames.

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import loadmat
import h5py

# Sampling frequency and window size
FS = 256
WINDOW_SAMPLES = FS  # one-second windows

# Directories
DATA_DIR = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts"
OUTPUT_DIR = os.path.join(os.getcwd(), 'cv_outputs')

# Helper functions

def load_data(file_path):
    """
    Load EEG data from .mat or .pkl into shape (channels, samples).
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
            # (epochs, times, channels)
            ep, tm, ch = loaded.shape
            data = loaded.transpose(2, 0, 1).reshape(ch, ep * tm)
        elif isinstance(loaded, list) and loaded and isinstance(loaded[0], dict) and 'eeg_wins' in loaded[0]:
            segments = []
            for trial in loaded:
                for win in trial['eeg_wins']:
                    arr = np.array(win)
                    segments.append(arr.T)
            data = np.hstack(segments)
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


def compute_plv_matrix(eeg_segment):
    """
    Compute PLV matrix for a segment of shape (channels, samples).
    Returns (channels, channels).
    """
    analytic = hilbert(eeg_segment, axis=1)
    phases = np.angle(analytic)
    n_ch = phases.shape[0]
    plv = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i, n_ch):
            diff = phases[i] - phases[j]
            val = np.abs(np.mean(np.exp(1j * diff)))
            plv[i, j] = plv[j, i] = val
    return plv


def compute_cv_map(plv_stack):
    """
    Given PLV stack of shape (n_windows, channels, channels), compute CV map.
    """
    mean = np.mean(plv_stack, axis=0)
    std = np.std(plv_stack, axis=0)
    cv = np.zeros_like(mean)
    mask = mean != 0
    cv[mask] = std[mask] / mean[mask]
    return cv


def main():
    conditions = {
        'nf': os.path.join(DATA_DIR, 'neurofeedback', 'nf_results', 'Subject_000', 'Session_001'),
        'training': os.path.join(DATA_DIR, 'training'),
        'online_real': os.path.join(DATA_DIR, 'online', 'collection_results', 'Subject_000', 'Session_001'),
        'online_poc': os.path.join(DATA_DIR, 'online', 'PoC', 'Subject_000', 'Session_001')
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cond, folder in conditions.items():
        for ext in ['mat', 'pkl']:
            for file_path in glob.glob(os.path.join(folder, f"*.{ext}")):
                name = os.path.splitext(os.path.basename(file_path))[0]
                # Load raw EEG data
                eeg = load_data(file_path)
                n_ch, n_samples = eeg.shape
                n_win = n_samples // WINDOW_SAMPLES

                # Build PLV stack
                plv_stack = np.zeros((n_win, n_ch, n_ch))
                for w in range(n_win):
                    seg = eeg[:, w*WINDOW_SAMPLES:(w+1)*WINDOW_SAMPLES]
                    plv_stack[w] = compute_plv_matrix(seg)

                # Compute CV map
                cv_map = compute_cv_map(plv_stack)

                # Save CV heatmap
                out_folder = os.path.join(OUTPUT_DIR, cond, ext)
                os.makedirs(out_folder, exist_ok=True)
                plt.figure(figsize=(6,6))
                vmax = np.nanpercentile(cv_map, 95)
                plt.imshow(cv_map, vmin=0, vmax=vmax, cmap='RdYlGn')
                plt.title(f"CV Map: {cond}_{ext}_{name}")
                plt.colorbar(label='CV')
                save_path = os.path.join(out_folder, f"cv_{cond}_{ext}_{name}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved CV map: {save_path}")

if __name__ == '__main__':
    main()
