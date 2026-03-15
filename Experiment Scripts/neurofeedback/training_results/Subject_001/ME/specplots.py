# -*- coding: utf-8 -*-
"""
EEG trial inspector:
  reduce -> zscore -> 8-30 Hz band-pass -> small Laplacian (C3/C4)
  -> spectrograms (ShortTimeFFT)
  -> AR PSD (Yule–Walker) + bandpower in a target band
  -> cursor_x time-series (if available)
Output:
  For each trial, one PNG with:
    - Row 1: cursor_x time series (full width)
    - Row 2: spectrogram (C3)  (full width)
    - Row 3: spectrogram (C4)  (full width)
      (Rows 1–3 share the same time axis)
    - Row 4: AR PSD curves side-by-side (C3 | C4)
Title includes Left/Right and Hit/Miss; header notes AR band stats and whether cursor trace is shown.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from scipy.signal import ShortTimeFFT, windows
import scipy.signal as sig
from scipy.linalg import toeplitz

# -----------------------------------------
# config: channels, mapping, and constants
# -----------------------------------------

FS_DEFAULT = 256  # fs should always be 256 for my recordings unless overridden per trial

# canonical channel list (10-20 style ordering that matches the raw array)
headset_electrodes = [
    'FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
    'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1', 'A2'
]

# the subset indices (0-based) into the above list; this must match the raw array order
headset = {
    'FC3': 18,
    'FC4': 22,
    'C5' : 26,
    'C3' : 27,
    'C1' : 28,
    'C2' : 30,
    'C4' : 31,
    'C6' : 32,
    'CP3': 36,
    'CP4': 40
}

# keep this deterministic order so downstream arrays/plots are consistent
target_chan_order = ['FC3','FC4','C5','C3','C1','C2','C4','C6','CP3','CP4']

# small Laplacian neighbors (classic 10–20)
lap_neighbors = {
    'C3': ['FC3', 'C1', 'CP3', 'C5'],
    'C4': ['FC4', 'C2', 'CP4', 'C6']
}

# -----------------------------------------
# utils / helper funcs
# -----------------------------------------

def as_trials_iterable(data_obj):
    """unify iteration for dict OR list of trials."""
    if isinstance(data_obj, dict):
        for k in sorted(data_obj.keys()):
            yield k, data_obj[k]
    else:
        for i, tr in enumerate(data_obj):
            yield i, tr

def zscore_per_channel(x, eps=1e-12):
    """z-score per channel over time for (n_ch, n_samp)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (x - mu) / sd

def butter_bandpass_sos(lo, hi, fs, order=4):
    """design SOS bandpass."""
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    return butter(order, [lo_n, hi_n], btype='bandpass', output='sos')

def apply_bandpass(x, fs, lo=8.0, hi=30.0, order=4):
    """zero-phase band-pass per channel for (n_ch, n_samp)."""
    sos = butter_bandpass_sos(lo, hi, fs, order=order)
    return sosfiltfilt(sos, x, axis=1)

def reduce_normalize_filter(eeg_full, fs, keep_order, idx_map, lo=8, hi=30):
    """
    1) reduce channels to keep_order
    2) z-score per channel
    3) band-pass (default 8–30 Hz; set hi=15 for strict mu)
    returns (reduced_filtered, kept_names)
    """
    chosen_indices, chosen_names = [], []
    for nm in keep_order:
        if nm in idx_map:
            chosen_indices.append(idx_map[nm])
            chosen_names.append(nm)

    reduced = eeg_full[np.array(chosen_indices), :]
    reduced_z = zscore_per_channel(reduced)
    reduced_filt = apply_bandpass(reduced_z, fs, lo=lo, hi=hi, order=4)
    return reduced_filt, chosen_names

def compute_small_laplacian(reduced_eeg, chan_names):
    """
    compute small Laplacian for C3/C4 from reduced_eeg (n_ch, n_samp).
    returns dict with keys 'C3_sLap', 'C4_sLap' (or None if not computable).
    """
    name_to_idx = {nm: i for i, nm in enumerate(chan_names)}
    out = {}
    for center in ['C3', 'C4']:
        if center not in name_to_idx:
            out[f'{center}_sLap'] = None
            continue
        avail = [nb for nb in lap_neighbors[center] if nb in name_to_idx]
        c_idx = name_to_idx[center]
        if len(avail) == 0:
            out[f'{center}_sLap'] = None
            continue
        center_sig = reduced_eeg[c_idx]
        neigh_mean = reduced_eeg[[name_to_idx[nm] for nm in avail]].mean(axis=0)
        out[f'{center}_sLap'] = center_sig - neigh_mean
    return out

# ---- modern spectrogram via ShortTimeFFT ----

def make_spectrogram_stfft(sig, fs, window='hann', nperseg=128, noverlap=96, scale_to='psd'):
    """
    compute spectrogram with SciPy's ShortTimeFFT (preferred over legacy).
    returns f (Hz), t (s), Sxx (power), with Sxx.shape == (len(f), len(t)).
    guards short trials by clamping nperseg/noverlap.
    """
    N = len(sig)
    if N == 0:
        return np.array([]), np.array([]), np.empty((0, 0))

    nperseg_eff = min(nperseg, N)
    noverlap_eff = min(noverlap, nperseg_eff - 1) if nperseg_eff > 1 else 0
    hop = max(1, nperseg_eff - noverlap_eff)

    win = windows.get_window(window, nperseg_eff, fftbins=True)
    SFT = ShortTimeFFT(win, hop=hop, fs=fs, scale_to=scale_to)

    Sxx = SFT.spectrogram(sig)   # (n_f, n_frames), PSD if scale_to='psd'
    f = SFT.f                    # (n_f,)
    t = SFT.t(N)                 # (n_frames,)

    if Sxx.ndim == 1:
        Sxx = Sxx[:, None]
    if t.ndim != 1:
        t = np.ravel(t)
    return f, t, Sxx

# ---- AR PSD helpers (Yule–Walker) ----

def yule_walker_ar(x: np.ndarray, order: int):
    """
    Yule–Walker AR(p) with biased autocorr.
    Returns:
      A: AR polynomial coeffs [1, -a1, -a2, ..., -ap]
      sigma2: driving-noise variance
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    if n == 0:
        return np.array([1.0]), 0.0
    # autocorrelation r[0..order]
    r = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(order + 1)])
    # guard degenerate r0 / nan
    if r[0] <= 0 or np.any(~np.isfinite(r)):
        return np.array([1.0]), 0.0
    R = toeplitz(r[:-1])            # Toeplitz from r0..r_{p-1}
    a = np.linalg.solve(R, r[1:])   # solve R a = r(1..p)
    sigma2 = r[0] - np.dot(a, r[1:])
    A = np.concatenate(([1.0], -a))
    return A, float(max(sigma2, 0.0))

def ar_psd_curve(x: np.ndarray, order: int, fs: float, worN: int = 2048):
    """
    Return (freqs, psd) for AR(p) model fitted to x via Yule–Walker.
    PSD is |sqrt(sigma2) / A(e^{jw})|^2 over [0, fs/2].
    """
    A, sigma2 = yule_walker_ar(x, order)
    if A.ndim != 1 or A.size == 0:
        return np.array([]), np.array([])
    freqs, h = sig.freqz(np.sqrt(max(sigma2, 1e-12)), A, worN=worN, fs=fs)
    psd = np.abs(h) ** 2
    return freqs, psd

def ar_bandpower(x: np.ndarray, order: int, f_lo: float, f_hi: float, fs: float, worN: int = 4096):
    """
    Integrate AR PSD over [f_lo, f_hi].
    """
    f, psd = ar_psd_curve(x, order=order, fs=fs, worN=worN)
    if f.size == 0:
        return 0.0
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], f[mask]))

# ---- plotting helpers ----

def _crop_freq(f, Sxx, fmax):
    if fmax is None:
        return f, Sxx
    keep = f <= fmax
    if not np.any(keep):
        return np.array([]), np.empty((0, Sxx.shape[1]))
    return f[keep], Sxx[keep, :]

def _make_mesh(t, f):
    return np.meshgrid(t, f)

def plot_trial_spectrogram_and_arpsd(
    trial_id, label, hit, fs,
    spec_dict,                # {'C3_spec': (f,t,Sxx) or None, 'C4_spec': (f,t,Sxx) or None}
    ar_results,               # {'C3': {'f','psd','bandpower'}, 'C4': {...}}, 'delta', 'band'
    out_path,
    fmax_spec=20.0,
    vmin=None, vmax=None,
    title_prefix="Small Laplacian",
    # --- new args for cursor plotting ---
    cursor_x=None,
    cursor_fs=None
):
    """
    Layout:
      Row 1: Cursor (full width)
      Row 2: Spectrogram C3 (full width)
      Row 3: Spectrogram C4 (full width)
        -> Rows 1–3 share the same time axis (x-limits aligned)
      Row 4: PSD C3 | PSD C4 (side-by-side)
    """
    # unpack spectrograms
    entry_C3 = spec_dict.get('C3_spec', None)
    entry_C4 = spec_dict.get('C4_spec', None)

    # prepare spectrogram panels
    panels = []
    for center, entry in [('C3', entry_C3), ('C4', entry_C4)]:
        if entry is None:
            panels.append((center, None, None, None, None))
            continue
        f, t, Sxx = entry
        if f.size == 0 or t.size == 0 or Sxx.size == 0:
            panels.append((center, None, None, None, None))
            continue
        f_c, Sxx_c = _crop_freq(f, Sxx, fmax_spec)
        if f_c.size == 0 or Sxx_c.size == 0:
            panels.append((center, None, None, None, None))
            continue
        Sxx_c = Sxx_c[:len(f_c), :len(t)]
        if Sxx_c.shape != (len(f_c), len(t)):
            panels.append((center, None, None, None, None))
            continue
        Sxx_db = 10.0 * np.log10(Sxx_c + 1e-20)
        panels.append((center, f_c, t, Sxx_c, Sxx_db))

    # robust shared color scale for spectrograms
    if vmin is None or vmax is None:
        vals = [Sdb for _, _, _, _, Sdb in panels if Sdb is not None]
        if vals:
            all_db = np.concatenate([v.ravel() for v in vals])
            lo = np.nanpercentile(all_db, 2)
            hi = np.nanpercentile(all_db, 98)
            if vmin is None: vmin = lo
            if vmax is None: vmax = hi
        else:
            if vmin is None: vmin = -120
            if vmax is None: vmax = 0

    # figure with 4 rows:
    #   r1,r2,r3 span both columns; r4 has two columns
    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[0.8, 1.15, 1.15, 1.0])

    # Row 1: Cursor (span 2 cols)
    ax_cursor = fig.add_subplot(gs[0, :])

    # Row 2: Spec C3 (span 2 cols)
    ax_s_c3 = fig.add_subplot(gs[1, :])

    # Row 3: Spec C4 (span 2 cols)
    ax_s_c4 = fig.add_subplot(gs[2, :])

    # Row 4: PSDs side-by-side
    ax_p_c3 = fig.add_subplot(gs[3, 0])
    ax_p_c4 = fig.add_subplot(gs[3, 1])

    # ----- compute a common time limit for rows 1–3 -----
    # cursor time length
    tmax_cursor = None
    if cursor_x is not None:
        cursor_x = np.asarray(cursor_x).squeeze()
        if cursor_x.ndim == 0:
            cursor_x = np.array([float(cursor_x)])
        if cursor_x.size > 0:
            cfs = float(cursor_fs) if (cursor_fs is not None) else float(fs)
            cfs = max(cfs, 1.0)
            tmax_cursor = (len(cursor_x) - 1) / cfs

    # spectrogram time length (use the latest available end time)
    tmax_spec = None
    for _, f_c, t, _, _ in panels:
        if f_c is not None and t is not None and len(t) > 0:
            t_end = float(t[-1])
            tmax_spec = t_end if (tmax_spec is None or t_end > tmax_spec) else tmax_spec

    # choose common Tmax among available sources
    tmax = None
    for cand in (tmax_cursor, tmax_spec):
        if cand is not None:
            tmax = cand if (tmax is None or cand > tmax) else tmax
    if tmax is None:
        tmax = 0.0  # fallback so set_xlim doesn't choke

    # --------- Row 1: cursor_x time-series ----------
    if (cursor_x is not None) and (cursor_x.size > 0):
        t_cursor = np.arange(len(cursor_x), dtype=float) / max(cfs, 1.0)
        ax_cursor.plot(t_cursor, cursor_x, linewidth=1.5)
        ax_cursor.set_title("Cursor X position")
        ax_cursor.set_ylabel("Cursor X")
        ax_cursor.margins(x=0.01)
        # optional 0-line if trace straddles zero
        try:
            if np.nanmin(cursor_x) < 0 < np.nanmax(cursor_x):
                ax_cursor.axhline(0.0, linestyle='--', linewidth=0.8, alpha=0.6)
        except Exception:
            pass
    else:
        ax_cursor.text(0.5, 0.5, "No cursor_x", ha='center', va='center', transform=ax_cursor.transAxes, alpha=0.7)

    ax_cursor.set_xlim(0.0, tmax)
    ax_cursor.set_xlabel("")  # I keep x-label on the lowest time-aligned panel
    ax_cursor.tick_params(labelbottom=False)

    # --------- Row 2 & 3: Spectrograms (stacked, time-aligned) ----------
    last_pcm = None

    # C3
    entry = panels[0]
    if entry[1] is not None:
        _, f_c, t, _, Sxx_db = entry
        T, F = _make_mesh(t, f_c)
        pcm = ax_s_c3.pcolormesh(T, F, Sxx_db, shading='auto', vmin=vmin, vmax=vmax)
        last_pcm = pcm
        ax_s_c3.set_title(f"{title_prefix} @ C3")
        ax_s_c3.set_ylabel('Frequency (Hz)')
        ax_s_c3.set_xlim(0.0, tmax)
        ax_s_c3.tick_params(labelbottom=False)
    else:
        ax_s_c3.text(0.5, 0.5, "No C3 spectrogram", ha='center', va='center', transform=ax_s_c3.transAxes, alpha=0.7)
        ax_s_c3.set_xlim(0.0, tmax)
        ax_s_c3.tick_params(labelbottom=False)

    # C4
    entry = panels[1]
    if entry[1] is not None:
        _, f_c, t, _, Sxx_db = entry
        T, F = _make_mesh(t, f_c)
        pcm = ax_s_c4.pcolormesh(T, F, Sxx_db, shading='auto', vmin=vmin, vmax=vmax)
        last_pcm = pcm
        ax_s_c4.set_title(f"{title_prefix} @ C4")
        ax_s_c4.set_ylabel('Frequency (Hz)')
        ax_s_c4.set_xlabel('Time (s)')  # bottom of the time-aligned block
        ax_s_c4.set_xlim(0.0, tmax)
    else:
        ax_s_c4.text(0.5, 0.5, "No C4 spectrogram", ha='center', va='center', transform=ax_s_c4.transAxes, alpha=0.7)
        ax_s_c4.set_xlabel('Time (s)')
        ax_s_c4.set_xlim(0.0, tmax)

    # one shared horizontal colorbar for both spectrograms
    if last_pcm is not None:
        cbar = fig.colorbar(
            last_pcm, ax=[ax_s_c3, ax_s_c4],
            orientation='horizontal', pad=0.10, aspect=50
        )
        cbar.set_label('Power (dB)')

    # --------- Row 4: AR PSD curves (side-by-side) ----------
    band = ar_results.get('band', (None, None))
    p3 = ar_results.get('C3', {}).get('bandpower', None)
    p4 = ar_results.get('C4', {}).get('bandpower', None)
    delta = ar_results.get('delta', None)

    # C3 PSD
    f3 = ar_results.get('C3', {}).get('f', np.array([]))
    psd3 = ar_results.get('C3', {}).get('psd', np.array([]))
    if f3.size > 0:
        ax_p_c3.plot(f3, 10*np.log10(psd3 + 1e-20), label='AR PSD (C3)')
        if band[0] is not None and band[1] is not None:
            ax_p_c3.axvspan(band[0], band[1], alpha=0.15, label=f"Band {band[0]:.1f}–{band[1]:.1f} Hz")
        if p3 is not None:
            ax_p_c3.text(0.01, 0.95, f"Bandpower: {p3:.3f}", transform=ax_p_c3.transAxes,
                         va='top', ha='left', fontsize=10)
    ax_p_c3.set_xlim(0, 40)
    ax_p_c3.set_xlabel('Frequency (Hz)')
    ax_p_c3.set_ylabel('PSD (dB)')
    ax_p_c3.set_title('AR PSD @ C3')
    ax_p_c3.legend(loc='upper right')

    # C4 PSD
    f4 = ar_results.get('C4', {}).get('f', np.array([]))
    psd4 = ar_results.get('C4', {}).get('psd', np.array([]))
    if f4.size > 0:
        ax_p_c4.plot(f4, 10*np.log10(psd4 + 1e-20), label='AR PSD (C4)')
        if band[0] is not None and band[1] is not None:
            ax_p_c4.axvspan(band[0], band[1], alpha=0.15, label=f"Band {band[0]:.1f}–{band[1]:.1f} Hz")
        if p4 is not None:
            ax_p_c4.text(0.01, 0.95, f"Bandpower: {p4:.3f}", transform=ax_p_c4.transAxes,
                         va='top', ha='left', fontsize=10)
    ax_p_c4.set_xlim(0, 40)
    ax_p_c4.set_xlabel('Frequency (Hz)')
    ax_p_c4.set_ylabel('PSD (dB)')
    ax_p_c4.set_title('AR PSD @ C4')
    ax_p_c4.legend(loc='upper right')

    # --------- header ----------
    label_name = "Left" if label == 0 else "Right" if label == 1 else str(label)
    hit_text = "Hit" if bool(hit) else "Miss"
    header = f"Trial {trial_id} | {label_name} | {hit_text} | fs={fs} Hz"
    if (p3 is not None) and (p4 is not None) and (delta is not None) and (band[0] is not None):
        header += f" | AR band {band[0]:.1f}–{band[1]:.1f} Hz: P3={p3:.3f}, P4={p4:.3f}, Δ={delta:.3f}"
    if (cursor_x is not None) and (cursor_x.size > 0):
        header += " | cursor trace shown"
    fig.suptitle(header, fontsize=14)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------
# main pipeline driver
# -----------------------------------------

def process_all_trials(
    data_obj,
    keep_order=target_chan_order,
    idx_map=headset,
    fig_dir='figs_mu_8to30',
    plot_freq_max=20.0,
    # STFFT params (balanced temporal detail)
    stft_nperseg=128,
    stft_noverlap=96,
    # AR params (match online defaults unless I override)
    ar_order: int = 12,
    ar_band: tuple = (10.5, 13.5),
    ar_worN: int = 4096
):
    """
    iterate all trials, do:
      - reduce -> z-score -> band-pass (default 8–30 Hz here)
      - small Laplacian at C3/C4
      - STFFT spectrograms for C3_sLap and C4_sLap
      - AR PSD (Yule–Walker) for C3_sLap and C4_sLap + bandpower in ar_band
      - one PNG per trial with 4-row layout (cursor, spec C3, spec C4, PSDs)
    returns list of dicts with processed arrays + spectrograms + AR PSD curves and bandpowers.
    """
    os.makedirs(fig_dir, exist_ok=True)
    processed = []

    max_needed_idx = max(idx_map.values())

    for trial_key, trial in as_trials_iterable(data_obj):
        if 'eeg' not in trial:
            raise ValueError(f"Trial {trial_key} missing 'eeg'.")

        eeg = np.asarray(trial['eeg'])
        fs  = int(trial.get('fs', FS_DEFAULT))
        label = int(trial.get('label', -1))
        hit = bool(trial.get('hit', False))

        # optional cursor inputs; default to 60 Hz if not provided
        cursor_x = trial.get('cursor_x', None)
        cursor_fs = trial.get('cursor_fs', 60)

        # shape checks
        if eeg.ndim != 2:
            raise ValueError(f"Trial {trial_key} 'eeg' must be 2D (ch x samples). Got {eeg.shape}.")
        if eeg.shape[0] <= max_needed_idx:
            raise ValueError(
                f"Trial {trial_key} has {eeg.shape[0]} channels, but idx_map expects up to index "
                f"{max_needed_idx}. Check channel order/mapping."
            )

        # 1) reduce + z + bandpass (I currently use 8–30 here; hi=15.0 if I want strict mu)
        reduced_filt, kept_names = reduce_normalize_filter(eeg, fs, keep_order, idx_map, lo=8, hi=30.0)

        # 2) small Laplacian at C3/C4
        laps = compute_small_laplacian(reduced_filt, kept_names)
        lap3 = laps.get('C3_sLap', None)
        lap4 = laps.get('C4_sLap', None)

        # 3) spectrograms for each Laplacian (ShortTimeFFT)
        spec_info = {}
        for center, sig_ in [('C3', lap3), ('C4', lap4)]:
            if sig_ is None:
                spec_info[f'{center}_spec'] = None
                continue
            f_sp, t_sp, Sxx = make_spectrogram_stfft(
                sig_, fs, window='hann',
                nperseg=stft_nperseg, noverlap=stft_noverlap, scale_to='psd'
            )
            spec_info[f'{center}_spec'] = (f_sp, t_sp, Sxx)

        # 4) AR PSD + bandpower
        ar_results = {}
        if lap3 is not None:
            f_ar3, psd_ar3 = ar_psd_curve(lap3, order=ar_order, fs=fs, worN=ar_worN)
            p3 = ar_bandpower(lap3, order=ar_order, f_lo=ar_band[0], f_hi=ar_band[1], fs=fs, worN=ar_worN)
            ar_results['C3'] = {'f': f_ar3, 'psd': psd_ar3, 'bandpower': p3}
        else:
            p3 = None
            ar_results['C3'] = {'f': np.array([]), 'psd': np.array([]), 'bandpower': None}

        if lap4 is not None:
            f_ar4, psd_ar4 = ar_psd_curve(lap4, order=ar_order, fs=fs, worN=ar_worN)
            p4 = ar_bandpower(lap4, order=ar_order, f_lo=ar_band[0], f_hi=ar_band[1], fs=fs, worN=ar_worN)
            ar_results['C4'] = {'f': f_ar4, 'psd': psd_ar4, 'bandpower': p4}
        else:
            p4 = None
            ar_results['C4'] = {'f': np.array([]), 'psd': np.array([]), 'bandpower': None}

        delta = (p4 - p3) if (p3 is not None and p4 is not None) else None
        ar_results['delta'] = delta
        ar_results['band'] = ar_band

        # 5) figure with cursor/top, specs stacked, PSDs bottom
        pair_path = os.path.join(fig_dir, f"trial_{int(trial_key):04d}_C3C4_sLap.png")
        plot_trial_spectrogram_and_arpsd(
            trial_id=trial_key,
            label=label,
            hit=hit,
            fs=fs,
            spec_dict=spec_info,
            ar_results=ar_results,
            out_path=pair_path,
            fmax_spec=plot_freq_max,
            vmin=None, vmax=None,
            title_prefix="Small Laplacian",
            cursor_x=cursor_x,
            cursor_fs=cursor_fs
        )

        # stash outputs for downstream/introspection
        processed.append({
            'trial_id': trial_key,
            'label': label,
            'hit': hit,
            'fs': fs,
            'kept_channel_names': kept_names,
            'eeg_reduced_z_bp': reduced_filt,
            'C3_small_laplacian': lap3,
            'C4_small_laplacian': lap4,
            'spectrograms': spec_info,
            'ar': {
                'order': ar_order, 'band': ar_band, 'delta': delta,
                'C3': ar_results['C3'],  # 'f', 'psd', 'bandpower'
                'C4': ar_results['C4'],
            },
            'cursor': {
                'x': None if cursor_x is None else np.asarray(cursor_x),
                'fs': cursor_fs if (cursor_fs is not None) else fs
            }
        })

    return processed

# -----------------------------------------
# example: run script
# -----------------------------------------
if __name__ == "__main__":
    # tweak this if my file lives elsewhere
    DATA_PATH = "session_data.pkl"
    FIG_DIR   = "figs_mu_8to30"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Couldn't find {DATA_PATH}. Update DATA_PATH or provide data object.")

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    results = process_all_trials(
        data,
        fig_dir=FIG_DIR,
        plot_freq_max=40.0,
        stft_nperseg=128,  # 0.5 s window @256 Hz
        stft_noverlap=96,  # 75% overlap → 125 ms hop
        ar_order=12,
        ar_band=(10.5, 13.5),
        ar_worN=4096
    )
    print(f"Processed {len(results)} trials. PNGs saved to '{FIG_DIR}/'.")
