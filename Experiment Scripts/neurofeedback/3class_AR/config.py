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

# ─── NO-CONTROL / BASELINE (AR) ─────────────────────────────────────────
# I collect an explicit baseline at the start of each level. During that
# rest block, I estimate mean/std of the AR evidence (log power ratio) so
# I can abstain (no-control) when the participant isn't producing clear MI.
REST_BASELINE_SEC = 30.0   # s (set 30–60s as needed)

# numerical guard for log/ratio calculations
AR_EPS = 1e-12

# I convert evidence to z-scores using the baseline mean/std. If |z| is below
# this threshold, I output NO_CONTROL (-1) and the cursor doesn't move.
# I only move when evidence is confidently away from baseline.
# z = (d - mu) / (sigma + eps), where d = log(P_C4) - log(P_C3)
AR_Z_THRESHOLD = 0.6

# ─── CSP CHANNELS  ──
CSP_CHANNELS = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
]

# ─── SUBJECT & SESSION ──────────────────────────────────────────────────
SUBJECT_ID  = "000"
SESSION_ID  = "001"
RESULTS_DIR = "./training_results"

SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR, f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)
