# training_pipeline.py

import numpy as np
import scipy.signal as sig
from scipy.linalg import toeplitz
from collections import deque

from config import SAMPLING_RATE, STEP_SIZE, AR_BASELINE_SECONDS, AR_ZSCORE_SECONDS, AR_ZSCORE_CLIP
from preprocess import Preprocessor


def yule_walker_ar(x: np.ndarray, order: int):
    """
    Yule–Walker AR estimation (biased autocorr).
    Returns:
      A: AR polynomial coeffs [1, -a1, -a2, ..., -ap]
      sigma2: driving-noise variance
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size

    # I compute autocorrelation r[0..order] (biased)
    r = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(order + 1)])
    R = toeplitz(r[:-1])            # Toeplitz matrix from r0..r_{p-1}
    a = np.linalg.solve(R, r[1:])   # solve R a = r(1..p)
    sigma2 = r[0] - np.dot(a, r[1:])  # prediction error variance
    A = np.concatenate(([1.0], -a))   # polynomial for freqz denominator
    return A, float(sigma2)


def ar_bandpower(x: np.ndarray, order: int, f_lo: float, f_hi: float, fs: float):
    """
    AR PSD via freq response of sqrt(sigma2)/A(z); integrate over f_lo..f_hi.
    """
    A, sigma2 = yule_walker_ar(x, order)

    # Frequency response
    freqs, h = sig.freqz(np.sqrt(max(sigma2, 1e-12)), A, worN=1024, fs=fs)
    psd = np.abs(h) ** 2

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0

    return float(np.trapz(psd[mask], freqs[mask]))


class ARPipeline:
    """
    Motor-imagery neurofeedback (BCI2000-style, LR-only):
      Hjorth Laplacian(C3/C4) → AR PSD → integrate band → diff (C4 - C3)
      baseline-correct + running z-score → binary LR command.

    Notes:
      - During the initial baseline period I return `None` (no movement).
      - After baseline, I z-score the baseline-corrected difference using a rolling window.
    """

    def __init__(
        self,
        band=(10.5, 13.5),
        order=12,
        baseline_seconds: float = AR_BASELINE_SECONDS,
        zscore_seconds: float = AR_ZSCORE_SECONDS,
        z_clip: float = AR_ZSCORE_CLIP,
        fs: float = SAMPLING_RATE,
        step_size: int = STEP_SIZE,
    ):
        self.band = band
        self.order = int(order)
        self.fs = float(fs)

        # I define how often the caller updates control (used to convert seconds → #updates)
        self.step_size = int(step_size)
        self.updates_per_sec = max(1.0, self.fs / max(1, self.step_size))

        self.baseline_updates = int(max(0.0, float(baseline_seconds)) * self.updates_per_sec)
        self.z_hist_len = int(max(1.0, float(zscore_seconds)) * self.updates_per_sec)
        self.z_clip = float(z_clip)

        # I keep baseline + rolling stats on the AR power difference
        self._baseline_buf = []  # I only need the mean, so list is fine for a short baseline
        self._baseline_mean = 0.0
        self._baseline_ready = (self.baseline_updates == 0)

        self._diff_hist = deque(maxlen=self.z_hist_len)

        # for logging / debugging if I want it
        self.last_diff = 0.0
        self.last_z = 0.0

        pp = Preprocessor()
        self.e = pp.headset_electrodes
        self.idx = {ch: i for i, ch in enumerate(self.e)}

        # canonical Hjorth 4-neighbour sets for motor electrodes
        self.hjorth_neighbors = {
            'C3': ['FC3', 'CP3', 'C1', 'C5'],
            'C4': ['FC4', 'CP4', 'C2', 'C6'],
        }

    def _laplacian_hjorth(self, win: np.ndarray, center: str, neighbors: list[str]) -> np.ndarray:
        """
        Hjorth Laplacian: center - mean(neighbors)
        - win: [samples, 64]
        I make this robust to missing channels by using whatever neighbours exist.
        """
        i = self.idx
        c = win[:, i[center]]

        avail = [win[:, i[ch]] for ch in neighbors if ch in i]
        if len(avail) == 0:
            return c

        neigh = np.stack(avail, axis=1)     # [samples, n_neigh]
        return c - neigh.mean(axis=1)       # center - mean(neighbours)

    def _update_baseline(self, diff: float) -> bool:
        """Return True if baseline is ready after updating."""
        if self._baseline_ready:
            return True

        self._baseline_buf.append(float(diff))
        if len(self._baseline_buf) >= self.baseline_updates:
            self._baseline_mean = float(np.mean(self._baseline_buf)) if self._baseline_buf else 0.0
            self._baseline_ready = True
        return self._baseline_ready

    def _zscore(self, x: float) -> float:
        """Rolling z-score using the recent history buffer."""
        self._diff_hist.append(float(x))
        arr = np.asarray(self._diff_hist, dtype=float)
    
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    
        # I use a scale-aware floor because my control signal can live at ~1e-10
        eps = 1e-12 + 1e-3 * abs(mu)
        if sd < eps:
            return 0.0
    
        z = (float(x) - mu) / sd
        if self.z_clip > 0:
            z = float(np.clip(z, -self.z_clip, self.z_clip))
        return z


    def process(self, window: np.ndarray):
        """
        window: [samples, 64] raw segment (caller provides the last ~2 s)
        returns:
          - None during baseline (no cursor movement)
          - 0 (move left) or 1 (move right) after baseline
        """
        # Hjorth 4-neighbour Laplacian for C3/C4
        lap3 = self._laplacian_hjorth(window, 'C3', self.hjorth_neighbors['C3'])
        lap4 = self._laplacian_hjorth(window, 'C4', self.hjorth_neighbors['C4'])

        # AR band power in my target band (defaults to ~mu)
        p3 = ar_bandpower(lap3, self.order, self.band[0], self.band[1], self.fs)
        p4 = ar_bandpower(lap4, self.order, self.band[0], self.band[1], self.fs)

        diff = float(p4 - p3)
        self.last_diff = diff

        # 1) baseline collection (rest)
        if not self._update_baseline(diff):
            self.last_z = 0.0
            return None  # I keep cursor centered while I estimate baseline

        # 2) baseline correction
        diff_b = diff - self._baseline_mean

        # 3) running z-score (zero mean / unit variance over rolling horizon)
        z = self._zscore(diff_b)
        self.last_z = z

        # 4) binary LR decision (BCI2000 used magnitude too; we only need direction)
        return 1 if z > 0 else 0
