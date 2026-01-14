import os

# -----------------------
# Session / output
# -----------------------
SUBJECT_ID  = "000"
SESSION_ID  = "000"
RESULTS_DIR = "./online_results"

SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR, f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)

# -----------------------
# Online inference timing
# -----------------------
SAMPLING_RATE     = 256          # must match your amp/LSL nominal rate (or what you resample to)
WINDOW_SEC        = 3.0          # sliding window length for PLV
HOP_SEC           = 0.04         # how often you run inference (AR-like update = 40 ms)

# -----------------------
# Preprocessing
# -----------------------
BP_LOW_HZ  = 8.0
BP_HIGH_HZ = 30.0
BP_ORDER   = 4

ARTIFACT_PTP_UV = 300.0          # adjust to your units; if raw is in uV, 300uV is sensible
ENABLE_ZSCORE   = False          # usually False for PLV (phase-based), but keep option

# -----------------------
# PLV graph construction
# -----------------------
TOPK_PERCENT = 0.40
EPSILON      = 1e-6

# -----------------------
# Model
# -----------------------
FOUNDATION_PT = "C:/Users/uceerjp/Desktop/PhD/Year 2/online experiments/Online-Motor-Imagery-Decoder/Experiment Scripts/neurofeedback/trained_models/foundational.pt"   
DEVICE_STR    = "cpu"  # "cuda" or "cpu"

# -----------------------
# Game
# -----------------------
NUM_LEVELS         = 10
TRIALS_PER_LEVEL   = 20
CUE_DURATION       = 1.0
TRIAL_DURATION     = 10.0
INTER_TRIAL_PAUSE  = 2.0
INTER_LEVEL_PAUSE  = 5.0

PLAYER_SPEED       = 2     # pixels/frame
