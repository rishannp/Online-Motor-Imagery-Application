# config.py  (CSP APP)
import os, glob

# ─── CSP APP ROOT ────────────────────────────────────────────────────────
CSP_APP_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\csp"

# ─── SUBJECT ─────────────────────────────────────────────────────────────
SUBJECT_ID = "002"

# ─── CURRENT LIVE SESSION (this run) ─────────────────────────────────────
CURRENT_SESSION_ID = "004"

# ─── APP ROOTS ───────────────────────────────────────────────────────────
_EXPERIMENT_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts"

_APP_TRAINING_ROOTS = {
    "neurofeedback": os.path.join(_EXPERIMENT_ROOT, "neurofeedback", "training_results"),
    "csp":           os.path.join(_EXPERIMENT_ROOT, "csp",           "training_results"),
    "graph_ml":      os.path.join(_EXPERIMENT_ROOT, "graph_ml",      "training_results"),
}

# ─── TRAINING SOURCE CONFIG ───────────────────────────────────────────────
# TRAIN_APPS: which apps to pull training data from.
#   Any subset of: "neurofeedback", "csp", "graph_ml"
TRAIN_APPS = ["neurofeedback"]

# TRAIN_MODE: which sessions to load from each app.
#   "last"  -> only the most recent Session_XXX folder for this subject
#   "all"   -> every Session_XXX folder found for this subject
TRAIN_MODE = "last"

# ─── SESSION RESOLVER ────────────────────────────────────────────────────
def _find_session_pkls(train_apps, train_mode, subject_id):
    """Return list of (app_name, session_id, abs_pkl_path), sorted chronologically within each app."""
    results = []
    for app_name in train_apps:
        if app_name not in _APP_TRAINING_ROOTS:
            raise ValueError(
                f"Unknown app '{app_name}' in TRAIN_APPS. "
                f"Must be one of: {list(_APP_TRAINING_ROOTS.keys())}"
            )
        subject_dir = os.path.join(_APP_TRAINING_ROOTS[app_name], f"Subject_{subject_id}")
        if not os.path.isdir(subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        session_dirs = sorted(
            d for d in glob.glob(os.path.join(subject_dir, "Session_*"))
            if os.path.isdir(d)
        )
        if not session_dirs:
            raise FileNotFoundError(f"No Session_* folders found under: {subject_dir}")

        if train_mode == "last":
            session_dirs = [session_dirs[-1]]
        elif train_mode == "all":
            pass
        else:
            raise ValueError(f"TRAIN_MODE must be 'last' or 'all', got: '{train_mode}'")

        for sd in session_dirs:
            session_id = os.path.basename(sd).replace("Session_", "")
            pkl_path   = os.path.join(sd, "session_data.pkl")
            results.append((app_name, session_id, pkl_path))

    if not results:
        raise RuntimeError("No training sessions resolved. Check TRAIN_APPS / TRAIN_MODE / subject directory.")
    return results


TRAIN_SESSION_PKLS = _find_session_pkls(TRAIN_APPS, TRAIN_MODE, SUBJECT_ID)

# Human-readable summary string used in logging and directory names
TRAIN_SESSION_ID = ", ".join(f"{a}:{s}" for a, s, _ in TRAIN_SESSION_PKLS)

# ─── CSP APP OUTPUT LOCATIONS ────────────────────────────────────────────
RESULTS_DIR = os.path.join(CSP_APP_ROOT, "training_results")
MODELS_DIR  = os.path.join(CSP_APP_ROOT, "trained_models")

CURRENT_SESSION_DIR = os.path.join(
    RESULTS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"Session_{CURRENT_SESSION_ID}",
)
os.makedirs(CURRENT_SESSION_DIR, exist_ok=True)

_train_tag = "_".join(f"{a}{s}" for a, s, _ in TRAIN_SESSION_PKLS)
MODEL_OUT_DIR = os.path.join(
    MODELS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"Train_{_train_tag}_for_Session_{CURRENT_SESSION_ID}",
)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ─── GAME PARAMETERS ─────────────────────────────────────────────────────
NUM_LEVELS        = 5
TRIALS_PER_LEVEL  = 20
INTER_TRIAL_PAUSE = 2.0
INTER_LEVEL_PAUSE = 5.0
CUE_DURATION      = 1.0
TRIAL_DURATION    = 10.0

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

CSP_SMOOTH_VOTES = 15

# ─── ONLINE BASELINE + MARGIN CONTROL ────────────────────────────────────
CSP_BASELINE_CENTER = True
CSP_ANALOG_OUTPUT   = False
CSP_CMD_EPS         = 0.05

# ─── LSL ─────────────────────────────────────────────────────────────────
LSL_STREAM_TYPE = "EEG"
LSL_TIMEOUT_SEC = 5.0