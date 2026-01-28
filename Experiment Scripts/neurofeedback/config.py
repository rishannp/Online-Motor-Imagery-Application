# config.py

import os

# ─── MODE (trainer) ─────────────────────────────────────────────────────
METHOD = 'ar'  # alpha-asymmetry neurofeedback trainer

# ─── GAME PARAMETERS ────────────────────────────────────────────────────
NUM_LEVELS       = 10
TRIALS_PER_LEVEL = 20
INTER_TRIAL_PAUSE = 2.0
INTER_LEVEL_PAUSE = 5.0


# ─── CUE & TRIAL TIMING ─────────────────────────────────────────────────
CUE_DURATION   = 1.0   # s: highlight target side (we keep it yellow the full trial)
TRIAL_DURATION = 10.0   # s: trial timeout, same as Stieger

# ─── FEEDBACK RATE & WINDOW ─────────────────────────────────────────────
SAMPLING_RATE     = 256            # Hz
FEEDBACK_INTERVAL = 0.04           # (pprox. 40 ms)
WINDOW_SIZE       = SAMPLING_RATE * 2  # 3 s window (used for stability)
STEP_SIZE         = int(FEEDBACK_INTERVAL * SAMPLING_RATE)

# ─── CSP CHANNELS  ──
CSP_CHANNELS = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
]

# ─── SUBJECT & SESSION ──────────────────────────────────────────────────
SUBJECT_ID  = "008"
SESSION_ID  = "001"
RESULTS_DIR = "./training_results"

SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR, f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)

# ─── AR CONTROL NORMALISATION ────────────────────────────
# I optionally run a short baseline period (rest) at the beginning of the session.
# During baseline collection the pipeline returns `None` so the cursor stays centered.
AR_BASELINE_SECONDS = 10.0     # s: how long I estimate baseline asymmetry (C4-C3)
# I z-score the baseline-corrected AR difference using a rolling window of recent updates.
AR_ZSCORE_SECONDS   = 30.0     # s: rolling stats horizon (mean/std) for control stability
AR_ZSCORE_CLIP      = 3.0      # I clip |z| to avoid extreme control spikes
