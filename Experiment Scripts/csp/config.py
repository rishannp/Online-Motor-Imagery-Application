# config.py  (CSP APP)
import os

# ─── CSP APP ROOT ────────────────────────────────────────────────────────
CSP_APP_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\csp"

# ─── SUBJECT ─────────────────────────────────────────────────────────────
SUBJECT_ID = "005"
# where we read pkl to train)
TRAIN_SESSION_ID = "001"
# session we’re running right now)
CURRENT_SESSION_ID = "000"

# ─── WHERE TRAINING PKL LIVES (NEUROFEEDBACK FOLDER) ─────────────────────
NEUROFEEDBACK_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback"
NEUROFEEDBACK_TRAINING_RESULTS_DIR = os.path.join(NEUROFEEDBACK_ROOT, "training_results")

TRAIN_SESSION_PKL = os.path.join(
    NEUROFEEDBACK_TRAINING_RESULTS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"Session_{TRAIN_SESSION_ID}",
    "session_data.pkl",
)

# ─── CSP APP OUTPUT LOCATIONS ────────────────────────────────────────────
# Current session data is written here (NOT to neurofeedback)
RESULTS_DIR = os.path.join(CSP_APP_ROOT, "training_results")
MODELS_DIR  = os.path.join(CSP_APP_ROOT, "trained_models")

# where we store this sessions data
CURRENT_SESSION_DIR = os.path.join(
    RESULTS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"Session_{CURRENT_SESSION_ID}",
)
os.makedirs(CURRENT_SESSION_DIR, exist_ok=True)

# Model outputs live in CSP app folder, keyed by the TRAIN session
MODEL_OUT_DIR = os.path.join(
    MODELS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"TrainSession_{TRAIN_SESSION_ID}_for_CurrentSession_{CURRENT_SESSION_ID}",
)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ─── GAME PARAMETERS ───────────────────────────────
NUM_LEVELS        = 1
TRIALS_PER_LEVEL  = 20
INTER_TRIAL_PAUSE = 2.0
INTER_LEVEL_PAUSE = 5.0

CUE_DURATION   = 1.0
TRIAL_DURATION = 10.0

SAMPLING_RATE     = 256
FEEDBACK_INTERVAL = 0.04
WINDOW_SIZE       = SAMPLING_RATE * 1
STEP_SIZE         = int(FEEDBACK_INTERVAL * SAMPLING_RATE)

# ─── CSP SETTINGS ────────────────────────────────────────────────────────
CSP_BASELINE_SECONDS = 10.0

APPLY_BANDPASS = True
BP_LO, BP_HI   = 8.0, 30.0
BP_ORDER       = 4

ENABLE_ZSCORE = False

CSP_NCOMP = 6
CSP_REG   = 1e-6

CSP_MOTOR_SUBSET = True
CSP_CHANNELS = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
]

CSP_SMOOTH_VOTES = 5

# ─── ONLINE BASELINE + MARGIN CONTROL ─────────────────────────────
# During the first CSP_BASELINE_SECONDS, we estimate a baseline SVM margin
# (rest state). After baseline, we subtract that mean so outputs are
# centered per-session (non-stationarity guard).
CSP_BASELINE_CENTER = True

# DISCRETE MODE: output 0/1 commands (fixed speed in game).
# Negative margin => LEFT (0), positive margin => RIGHT (1).
CSP_ANALOG_OUTPUT = False

# Kept for compatibility; not used in discrete mode.
CSP_CMD_EPS = 0.05

# ─── LSL ────────────────────────────────────────────────────────────────
LSL_STREAM_TYPE  = "EEG"
LSL_TIMEOUT_SEC  = 5.0
