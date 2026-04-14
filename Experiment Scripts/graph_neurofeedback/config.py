# config.py

import os

# ─── MODE ───────────────────────────────────────────────────────────────
METHOD = 'ar'  # used by preprocess.py; PLV pipeline bypasses it directly

# ─── GAME PARAMETERS ───────────────────────────────────────────────────
NUM_LEVELS        = 5
TRIALS_PER_LEVEL  = 20
INTER_TRIAL_PAUSE = 2.0
INTER_LEVEL_PAUSE = 5.0

# ─── CUE & TRIAL TIMING ─────────────────────────────────────────────────
CUE_DURATION   = 1.0
TRIAL_DURATION = 10.0

# ─── FEEDBACK RATE & WINDOW ─────────────────────────────────────────────
SAMPLING_RATE     = 256
FEEDBACK_INTERVAL = 0.04
WINDOW_SIZE       = SAMPLING_RATE * 2   # 2 s window for PLV/ImCoh stability
STEP_SIZE         = int(FEEDBACK_INTERVAL * SAMPLING_RATE)

# ─── CSP CHANNELS ─────────────────
CSP_CHANNELS = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
]

# ─── PLV MOTOR ASYMMETRY CHANNELS ────────────────────────────────────────
# Midline (Cz, FCz) excluded — not lateralised.
PLV_LEFT_CHANNELS = [
    'FC5', 'FC3', 'FC1',
    'C5',  'C3',  'C1',
    'CP5', 'CP3', 'CP1',
]
PLV_RIGHT_CHANNELS = [
    'FC6', 'FC4', 'FC2',
    'C6',  'C4',  'C2',
    'CP6', 'CP4', 'CP2',
]

# ─── CONNECTIVITY METHOD ─────────────────────────────────────────────────
# Controls which connectivity measure is used inside PLVAsymmetryPipeline.
#
#   "plv"   — Phase Locking Value.
#             Sensitive to volume conduction (zero-lag coupling inflates scores).
#             Simpler and faster to interpret.
#
#   "imcoh" — Imaginary part of Coherency.
#             Immune to volume conduction by construction — zero-lag coupling
#             (field spread) contributes nothing to the imaginary part.
#
PLV_METHOD = "imcoh"  # "plv" or "imcoh"

# ─── PLV PIPELINE CONTROL ────────────────────────────────────────────────
PLV_BASELINE_SECONDS = 10.0
PLV_ZSCORE_SECONDS   = 30.0
PLV_ZSCORE_CLIP      = 3.0
PLV_BANDPASS_LO      = 8.0
PLV_BANDPASS_HI      = 30.0
PLV_BANDPASS_ORDER   = 4

# ─── SUBJECT & SESSION ──────────────────────────────────────────────────
SUBJECT_ID  = "012"
SESSION_ID  = "001"
RESULTS_DIR = "./training_results"

SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR,  f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)

# ─── AR CONTROL (kept for main_training.py compatibility) ────────────────
AR_BASELINE_SECONDS = 10.0
AR_ZSCORE_SECONDS   = 30.0
AR_ZSCORE_CLIP      = 3.0
