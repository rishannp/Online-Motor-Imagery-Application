# plv_pipeline.py
#
# Motor-asymmetry neurofeedback pipeline supporting two connectivity measures,
# selected by PLV_METHOD in config.py:
#
#   "plv"   — Phase Locking Value
#             PLV[i,j] = |mean(exp(i*(phi_i - phi_j)))|
#             Values in [0, 1]. Sensitive to volume conduction because
#             zero-lag coupling (field spread) produces PLV = 1.
#
#   "imcoh" — Imaginary part of Coherency
#             ImCoh[i,j] = Im(Cxy) / sqrt(Cxx * Cyy)
#             where Cxy = mean(conj(a_i) * a_j) over the window.
#             Zero-lag coupling => purely real cross-spectrum => ImCoh = 0.
#             Immune to volume conduction by construction.
#             Values in [-1, 1]; sign encodes phase lead/lag direction.
#             We use |ImCoh| as the connectivity strength (sign carries
#             lead/lag info, not lateralisation info).
#
# Asymmetry score (same structure for both measures):
#
#   intra_left  = mean connectivity of left-left electrode pairs
#   intra_right = mean connectivity of right-right electrode pairs
#   cross       = mean connectivity of left-right electrode pairs
#
#   raw_asym = (intra_right - intra_left) / (cross + eps)
#
# Positive => right hemisphere dominant => RIGHT (1)
# Negative => left  hemisphere dominant => LEFT  (0)

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from collections import deque

from config import (
    SAMPLING_RATE, STEP_SIZE,
    PLV_LEFT_CHANNELS, PLV_RIGHT_CHANNELS,
    PLV_BASELINE_SECONDS, PLV_ZSCORE_SECONDS, PLV_ZSCORE_CLIP,
    PLV_BANDPASS_LO, PLV_BANDPASS_HI, PLV_BANDPASS_ORDER,
    PLV_METHOD,
)

_HEADSET_64 = (
    "FP1 FPz FP2 AF7 AF3 AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 "
    "FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 T7 C5 C3 C1 Cz C2 C4 C6 "
    "T8 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 "
    "PO7 PO3 POz PO4 PO8 O1 Oz O2 F9 F10 A1 A2"
).split()

_CH_IDX = {ch: i for i, ch in enumerate(_HEADSET_64)}


def _make_bandpass(lo, hi, order, fs):
    nyq = 0.5 * float(fs)
    return butter(int(order), [float(lo) / nyq, float(hi) / nyq], btype="band")


def _compute_plv_matrix(analytic_tc: np.ndarray) -> np.ndarray:
    """
    analytic_tc: [T, C] complex analytic signal.
    Returns PLV matrix [C, C] in [0, 1].
    PLV[i,j] = |mean(exp(i*(phi_i - phi_j)))| = |mean(conj(u_i) * u_j)|
    where u = exp(i*phase), so |u| = 1.
    """
    phase = np.angle(analytic_tc)
    u     = np.exp(1j * phase)          # [T, C], unit-magnitude phasors
    T     = u.shape[0]
    plv   = np.abs((u.conj().T @ u) / max(1, T)).astype(np.float32)
    np.fill_diagonal(plv, 1.0)
    return plv


def _compute_imcoh_matrix(analytic_tc: np.ndarray) -> np.ndarray:
    """
    analytic_tc: [T, C] complex analytic signal.
    Returns |ImCoh| matrix [C, C] in [0, 1].

    ImCoh[i,j] = Im(Cxy) / sqrt(Cxx * Cyy)
    where Cxy = (1/T) * sum_t( conj(a_i(t)) * a_j(t) )

    Taking the absolute value discards phase lead/lag direction and
    retains only connectivity strength — appropriate for asymmetry scoring.
    Zero-lag coupling (volume conduction) => Cxy purely real => ImCoh = 0.
    """
    T  = analytic_tc.shape[0]
    # Cross-spectral matrix (averaged over time, single broadband estimate)
    C  = analytic_tc.shape[1]
    cs = (analytic_tc.conj().T @ analytic_tc) / max(1, T)  # [C, C] complex

    # Auto-power per channel (real, positive)
    auto = np.real(np.diag(cs))                             # [C]
    denom = np.sqrt(np.outer(auto, auto)) + 1e-12           # [C, C]

    imcoh = np.abs(np.imag(cs) / denom).astype(np.float32)  # [C, C], in [0,1]
    np.fill_diagonal(imcoh, 0.0)  # self-connectivity undefined; set to 0
    return imcoh


def _connectivity_matrix(analytic_tc: np.ndarray, method: str) -> np.ndarray:
    """Dispatch to PLV or ImCoh based on method string."""
    if method == "plv":
        return _compute_plv_matrix(analytic_tc)
    elif method == "imcoh":
        return _compute_imcoh_matrix(analytic_tc)
    else:
        raise ValueError(f"PLV_METHOD must be 'plv' or 'imcoh', got: '{method}'")


class PLVAsymmetryPipeline:
    """
    Online motor-asymmetry decoder using PLV or ImCoh (set by PLV_METHOD).

    Interface identical to ARPipeline:
      process(window_t64) -> None (baseline) | 0 (LEFT) | 1 (RIGHT)

    Debug attributes (updated every process() call):
      last_intra_left  : float — mean connectivity within left hemisphere
      last_intra_right : float — mean connectivity within right hemisphere
      last_cross       : float — mean connectivity across hemispheres
      last_asym        : float — raw normalised asymmetry score
      last_z           : float — baseline-corrected, z-scored output
      method           : str   — which measure is active ("plv" or "imcoh")
    """

    def __init__(
        self,
        fs               = SAMPLING_RATE,
        step_size        = STEP_SIZE,
        baseline_seconds = PLV_BASELINE_SECONDS,
        zscore_seconds   = PLV_ZSCORE_SECONDS,
        z_clip           = PLV_ZSCORE_CLIP,
        bp_lo            = PLV_BANDPASS_LO,
        bp_hi            = PLV_BANDPASS_HI,
        bp_order         = PLV_BANDPASS_ORDER,
        left_channels    = PLV_LEFT_CHANNELS,
        right_channels   = PLV_RIGHT_CHANNELS,
        method           = PLV_METHOD,
    ):
        self.fs        = float(fs)
        self.step_size = int(step_size)
        self.z_clip    = float(z_clip)
        self.method    = str(method).lower()

        if self.method not in ("plv", "imcoh"):
            raise ValueError(f"PLV_METHOD must be 'plv' or 'imcoh', got: '{self.method}'")

        self._b, self._a = _make_bandpass(bp_lo, bp_hi, bp_order, fs)

        # Resolve channel indices
        all_motor = list(left_channels) + list(right_channels)
        missing = [ch for ch in all_motor if ch not in _CH_IDX]
        if missing:
            raise ValueError(f"Motor channels not in 64-ch layout: {missing}")

        self._motor_idx = np.array([_CH_IDX[ch] for ch in all_motor], dtype=int)
        self._n_left    = len(left_channels)
        self._n_right   = len(right_channels)

        # Precompute index masks into the [n_motor x n_motor] connectivity matrix
        nl, nr = self._n_left, self._n_right
        n      = nl + nr

        ll_r, ll_c = np.triu_indices(nl, k=1)           # left-left upper triangle
        rr_r, rr_c = np.triu_indices(nr, k=1)           # right-right upper triangle
        rr_r, rr_c = rr_r + nl, rr_c + nl               # offset into full matrix
        cr_r = np.repeat(np.arange(nl), nr)              # all left rows
        cr_c = np.tile(np.arange(nl, n), nl)             # all right cols

        self._ll = (ll_r, ll_c)
        self._rr = (rr_r, rr_c)
        self._cr = (cr_r, cr_c)

        # Baseline
        updates_per_sec      = max(1.0, self.fs / max(1, self.step_size))
        self._baseline_n     = int(max(0.0, float(baseline_seconds)) * updates_per_sec)
        self._baseline_buf   = []
        self._baseline_mean  = 0.0
        self._baseline_ready = (self._baseline_n == 0)

        # Rolling z-score
        z_hist_len      = int(max(1.0, float(zscore_seconds)) * updates_per_sec)
        self._diff_hist = deque(maxlen=z_hist_len)

        # Debug attributes
        self.last_intra_left  = 0.0
        self.last_intra_right = 0.0
        self.last_cross       = 0.0
        self.last_asym        = 0.0
        self.last_z           = 0.0

    def _bandpass(self, window_tc: np.ndarray) -> np.ndarray:
        return filtfilt(self._b, self._a, window_tc, axis=0).astype(np.float32)

    def _asymmetry(self, window_t64: np.ndarray) -> float:
        # Extract motor channels and bandpass
        motor    = window_t64[:, self._motor_idx].astype(np.float32)  # [T, n_motor]
        motor    = self._bandpass(motor)

        # Analytic signal (shared by both PLV and ImCoh)
        analytic = hilbert(motor, axis=0)  # [T, n_motor] complex

        conn = _connectivity_matrix(analytic, self.method)  # [n_motor, n_motor]

        intra_left  = float(conn[self._ll].mean()) if self._ll[0].size > 0 else 0.0
        intra_right = float(conn[self._rr].mean()) if self._rr[0].size > 0 else 0.0
        cross       = float(conn[self._cr].mean()) if self._cr[0].size > 0 else 1.0

        self.last_intra_left  = intra_left
        self.last_intra_right = intra_right
        self.last_cross       = cross

        asym = (intra_right - intra_left) / (cross + 1e-6)
        self.last_asym = asym
        return asym

    def _update_baseline(self, asym: float) -> bool:
        if self._baseline_ready:
            return True
        self._baseline_buf.append(asym)
        if len(self._baseline_buf) >= self._baseline_n:
            self._baseline_mean  = float(np.mean(self._baseline_buf)) if self._baseline_buf else 0.0
            self._baseline_ready = True
        return self._baseline_ready

    def _zscore(self, x: float) -> float:
        self._diff_hist.append(float(x))
        arr = np.asarray(self._diff_hist, dtype=float)
        mu  = float(arr.mean())
        sd  = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        eps = 1e-12 + 1e-3 * abs(mu)
        if sd < eps:
            return 0.0
        z = (float(x) - mu) / sd
        if self.z_clip > 0:
            z = float(np.clip(z, -self.z_clip, self.z_clip))
        return z

    def process(self, window_t64: np.ndarray):
        """
        window_t64 : [T, 64] raw EEG window.
        Returns:
          None — baseline still running (cursor stays centred)
          0    — LEFT command
          1    — RIGHT command
        """
        asym = self._asymmetry(window_t64)

        if not self._update_baseline(asym):
            self.last_z = 0.0
            return None

        asym_c = asym - self._baseline_mean
        z      = self._zscore(asym_c)
        self.last_z = z

        return 1 if z > 0.0 else 0
