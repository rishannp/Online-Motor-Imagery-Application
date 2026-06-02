"""
Validate electrode montage by checking alpha-band topography.
Selects 5 random sessions (from eegGH + eegLS), one odd run per session.
Produces two plots per file:
  1. Broadband PSD waterfall ordered front-to-back
  2. Alpha-band topoplot
"""

import os, re, random
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
from scipy.interpolate import griddata

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIRS = [
    r"C:\Users\uceerjp\Desktop\G-W Data\eegGH",
    r"C:\Users\uceerjp\Desktop\G-W Data\eegLS",
]
LOCS_FILE = (
    r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments"
    r"\Online-Motor-Imagery-Decoder\EEGLabLocsMPICap.mat"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "montage_validation")
os.makedirs(OUT_DIR, exist_ok=True)

FS = 500          # Hz
N_EEG = 124       # first 124 channels are EEG
ALPHA_LO, ALPHA_HI = 8, 13  # Hz

# ── Load channel locations ──────────────────────────────────────────────────
def load_chanlocs(path):
    mat = sio.loadmat(path)
    cl = mat['Chanlocs'][0]
    labels = [str(cl[i]['labels'][0]) for i in range(len(cl))]
    # EEGLab convention: X = right, Y = anterior, Z = up
    x = np.array([float(cl[i]['X'][0][0]) for i in range(len(cl))])
    y = np.array([float(cl[i]['Y'][0][0]) for i in range(len(cl))])
    z = np.array([float(cl[i]['Z'][0][0]) for i in range(len(cl))])
    return labels, x, y, z

# ── File selection ──────────────────────────────────────────────────────────
def collect_sessions():
    """Return dict: session_id -> list of (filepath, run_num) for odd runs."""
    pat = re.compile(r'^(GH|LS)S(\d+)R(\d+)\.mat$', re.IGNORECASE)
    sessions = {}
    for d in DATA_DIRS:
        for fname in os.listdir(d):
            m = pat.match(fname)
            if not m:
                continue
            prefix, sess_str, run_str = m.group(1), m.group(2), m.group(3)
            run = int(run_str)
            if run % 2 == 0:   # skip even runs (task); keep odd (resting)
                continue
            sid = f"{prefix.upper()}S{sess_str}"
            full = os.path.join(d, fname)
            sessions.setdefault(sid, []).append((full, run))
    return sessions

def pick_files(n=5, seed=42):
    random.seed(seed)
    sessions = collect_sessions()
    chosen_sessions = random.sample(sorted(sessions.keys()), min(n, len(sessions)))
    files = []
    for sid in chosen_sessions:
        candidates = sessions[sid]
        path, run = random.choice(candidates)
        files.append((sid, run, path))
        print(f"  Selected: {os.path.basename(path)}")
    return files

# ── Signal helpers ──────────────────────────────────────────────────────────
def load_eeg(path):
    mat = sio.loadmat(path)
    sig = mat['signal'].astype(np.float64)  # (samples, 130)
    return sig[:, :N_EEG].T                 # (124, samples)

def compute_psd(data, fs=FS, nperseg=4*FS):
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd  # (124, freq_bins)

def alpha_power(freqs, psd):
    mask = (freqs >= ALPHA_LO) & (freqs <= ALPHA_HI)
    return psd[:, mask].mean(axis=1)  # (124,)

# ── Topoplot helpers ────────────────────────────────────────────────────────
def _cart_to_head(x, y, z):
    """Project 3-D Cartesian coordinates to 2-D head plane (azimuthal equidist.)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r == 0, 1e-9, r)
    # elevation angle from top
    elevation = np.arctan2(np.sqrt(x**2 + y**2), z)
    azimuth = np.arctan2(x, y)
    rho = elevation / (np.pi / 2)   # 0 = top, 1 = equator
    hx = rho * np.sin(azimuth)
    hy = rho * np.cos(azimuth)
    return hx, hy

def draw_topoplot(ax, values, hx, hy, labels=None, title=""):
    # Grid interpolation
    res = 300
    xi = np.linspace(-1.1, 1.1, res)
    yi = np.linspace(-1.1, 1.1, res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((hx, hy), values, (Xi, Yi), method='cubic')

    # Mask outside head circle
    mask = np.sqrt(Xi**2 + Yi**2) > 1.0
    Zi[mask] = np.nan

    # Use robust percentiles of the electrode VALUES (not grid) to set scale
    vmin = np.percentile(values, 5)
    vmax = np.percentile(values, 95)
    im = ax.contourf(Xi, Yi, Zi, levels=64, cmap='RdYlBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(Xi, Yi, Zi, levels=8, colors='k', linewidths=0.3, alpha=0.4)

    # Head circle + nose + ears
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k', lw=1.5)
    nose_x = [-.08, 0, .08]; nose_y = [.98, 1.12, .98]
    ax.plot(nose_x, nose_y, 'k', lw=1.5)
    # left ear
    ax.plot([-1.0, -1.08, -1.08, -1.0], [0.08, 0.04, -0.04, -0.08], 'k', lw=1.5)
    # right ear
    ax.plot([1.0, 1.08, 1.08, 1.0], [0.08, 0.04, -0.04, -0.08], 'k', lw=1.5)

    ax.scatter(hx, hy, s=8, c='k', zorder=5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)
    return im

# ── PSD waterfall (front → back) ────────────────────────────────────────────
def plot_psd_ordered(ax, freqs, psd, hy, labels, title=""):
    """Plot individual PSDs stacked, ordered front (top) to back (bottom)."""
    order = np.argsort(hy)[::-1]  # hy = anterior coord; descending = F→O
    freq_mask = freqs <= 50
    f = freqs[freq_mask]
    cmap = plt.cm.viridis
    n = len(order)
    for rank, idx in enumerate(order):
        color = cmap(rank / n)
        p_db = 10 * np.log10(psd[idx, freq_mask] + 1e-30)
        # offset each trace slightly so they don't overlap
        offset = -rank * 0.4
        ax.plot(f, p_db + offset, color=color, lw=0.5, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=8)
    ax.set_ylabel('PSD (dB) + offset (front→back)', fontsize=8)
    ax.set_title(title, fontsize=9)

    # Shade alpha band
    ax.axvspan(ALPHA_LO, ALPHA_HI, alpha=0.15, color='gold', label='Alpha')
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=7)

    # Colorbar-like label strip
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Front -> Back', fontsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Front', 'Back'])

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading channel locations...")
    labels, cx, cy, cz = load_chanlocs(LOCS_FILE)
    hx, hy = _cart_to_head(cx, cy, cz)

    print("\nSelecting files...")
    files = pick_files(n=5, seed=42)

    for sid, run, path in files:
        fname = os.path.basename(path)
        print(f"\nProcessing {fname} ...")
        data = load_eeg(path)
        freqs, psd = compute_psd(data)
        alpha_pw = alpha_power(freqs, psd)
        alpha_db = 10 * np.log10(alpha_pw + 1e-30)

        fig = plt.figure(figsize=(16, 7))
        fig.suptitle(f"Montage Validation — {fname}", fontsize=12, fontweight='bold')
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.35)

        ax_psd = fig.add_subplot(gs[0])
        ax_topo = fig.add_subplot(gs[1])

        # 1. Broadband PSD ordered front → back
        plot_psd_ordered(ax_psd, freqs, psd, hy, labels,
                         title="Broadband PSD (front → back, each trace = 1 electrode)")

        # 2. Alpha topoplot
        im = draw_topoplot(ax_topo, alpha_db, hx, hy, labels=labels,
                           title=f"Alpha power ({ALPHA_LO}–{ALPHA_HI} Hz, dB)\n"
                                 "Expected: posterior peak → correct montage")
        cbar2 = fig.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04)
        cbar2.set_label('dB', fontsize=8)
        cbar2.ax.tick_params(labelsize=7)

        out_path = os.path.join(OUT_DIR, fname.replace('.mat', '_montage_check.png'))
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved -> {out_path}")

    print(f"\nDone. All figures in: {OUT_DIR}")

if __name__ == '__main__':
    main()
