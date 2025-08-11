# training_pipeline.py

import numpy as np
import scipy.signal as sig
from scipy.linalg import toeplitz
from config import SAMPLING_RATE
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
      small-Laplacian(C3/C4) → AR(16) PSD → integrate 10.5–13.5 Hz → diff (C4 - C3)
      outputs a binary left/right command every ~40 ms.
    """
    def __init__(self, band=(10.5, 13.5), order=16):
        self.band = band
        self.order = order
        pp = Preprocessor()
        # Build a direct index lookup on the 64-ch naming you provided
        self.e = pp.headset_electrodes
        self.idx = {ch: i for i, ch in enumerate(self.e)}

    def _laplacian(self, win: np.ndarray, center: str, n1: str, n2: str) -> np.ndarray:
        i = self.idx
        return win[:, i[center]] - 0.5 * (win[:, i[n1]] + win[:, i[n2]])

    def process(self, window: np.ndarray) -> int:
        """
        window: [samples, 64] raw segment (we use last 3 s via caller)
        returns: 0 (move left) or 1 (move right)
        """
        # Small-Laplacian at motor electrodes
        lap3 = self._laplacian(window, 'C3', 'FC3', 'CP3')
        lap4 = self._laplacian(window, 'C4', 'FC4', 'CP4')

        # AR band power around 12 Hz (3 Hz bin)
        p3 = ar_bandpower(lap3, self.order, self.band[0], self.band[1], SAMPLING_RATE)
        p4 = ar_bandpower(lap4, self.order, self.band[0], self.band[1], SAMPLING_RATE)

        # Positive difference → RIGHT (1), else LEFT (0)
        return 1 if (p4 - p3) > 0 else 0
