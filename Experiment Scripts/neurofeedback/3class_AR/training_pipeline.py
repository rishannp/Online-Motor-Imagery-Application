# training_pipeline.py

import numpy as np
import scipy.signal as sig
from scipy.linalg import toeplitz
from config import SAMPLING_RATE, AR_Z_THRESHOLD, AR_EPS
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
    # autocorrelation r[0..order]
    r = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(order+1)])
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
    psd = np.abs(h)**2
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))

class ARPipeline:
    """
    Motor-imagery neurofeedback:
      Hjorth Laplacian(C3/C4) → AR PSD → integrate band → diff (C4 - C3)
      outputs a binary left/right command every ~40 ms.
    """
    def __init__(self, band=(10.5, 13.5), order=12):
        self.band = band
        self.order = order
        pp = Preprocessor()
        # Build a direct index lookup on the 64-ch naming you provided
        self.e = pp.headset_electrodes
        self.idx = {ch: i for i, ch in enumerate(self.e)}

        # define the canonical Hjorth 4-neighbour sets for motor electrodes
        self.hjorth_neighbors = {
            'C3': ['FC3', 'CP3', 'C1', 'C5'],
            'C4': ['FC4', 'CP4', 'C2', 'C6'],
        }

        # baseline params for no-control gating
        self._mu0 = None
        self._sd0 = None

    def set_baseline(self, mu: float, sd: float):
        # I store baseline mean/std so I can z-score evidence during the level
        self._mu0 = float(mu)
        self._sd0 = float(max(sd, 1e-9))

    def reset_baseline(self):
        # I clear baseline so I don't accidentally use stale values
        self._mu0 = None
        self._sd0 = None

    def evidence(self, window: np.ndarray) -> float:
        """Return a scalar MI evidence value from the current window.

        I use log power ratio to reduce scale sensitivity:
          d = log(p4 + eps) - log(p3 + eps)
        """
        lap3 = self._laplacian_hjorth(window, 'C3', self.hjorth_neighbors['C3'])
        lap4 = self._laplacian_hjorth(window, 'C4', self.hjorth_neighbors['C4'])

        p3 = ar_bandpower(lap3, self.order, self.band[0], self.band[1], SAMPLING_RATE)
        p4 = ar_bandpower(lap4, self.order, self.band[0], self.band[1], SAMPLING_RATE)
        return float(np.log(p4 + AR_EPS) - np.log(p3 + AR_EPS))

    def _laplacian_hjorth(self, win: np.ndarray, center: str, neighbors: list[str]) -> np.ndarray:
        """
        Hjorth Laplacian: center - mean(neighbors)
        - win: [samples, 64]
        - center: electrode name (e.g., 'C3')
        - neighbors: list of 4 nearest neighbours around 'center'
        I make this robust to missing channels by using whatever neighbours exist.
        """
        i = self.idx
        # center signal (this KeyError should never happen with the fixed 64-ch layout)
        c = win[:, i[center]]

        # collect neighbour columns that exist in the montage
        avail = [win[:, i[ch]] for ch in neighbors if ch in i]
        if len(avail) == 0:
            # degenerate case: no neighbours present — I just return the center signal
            return c

        neigh = np.stack(avail, axis=1)            # [samples, n_neigh]
        return c - neigh.mean(axis=1)              # center - mean(neighbours)

    def process(self, window: np.ndarray) -> int:
        """Return a command for the game.

        Outputs:
          -1 = NO_CONTROL (abstain)
           0 = LEFT
           1 = RIGHT
        """
        d = self.evidence(window)

        # If baseline isn't set yet, I refuse to emit forced left/right.
        if self._mu0 is None or self._sd0 is None:
            return -1

        z = (d - self._mu0) / (self._sd0 + AR_EPS)

        # deadband around baseline: only move when evidence is confidently away from rest
        if abs(z) < float(AR_Z_THRESHOLD):
            return -1

        return 1 if z > 0 else 0
