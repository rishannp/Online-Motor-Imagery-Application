# -*- coding: utf-8 -*-

# config.py  (GRAPH+ML APP)
import os

# ─── APP ROOT ────────────────────────────────────────────────────────────
GRAPH_APP_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\graph_ml"

# ─── SUBJECT ─────────────────────────────────────────────────────────────
SUBJECT_ID = "005"
# ─── TRAINING SOURCE SESSION (read pkl from neurofeedback) ───────────────
TRAIN_SESSION_ID = "001"
# ─── CURRENT LIVE SESSION (this run) ─────────────────────────────────────
CURRENT_SESSION_ID = "000"

# ─── WHERE TRAINING PKL LIVES (NEUROFEEDBACK) ────────────────────────────
NEUROFEEDBACK_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback"
NEUROFEEDBACK_TRAINING_RESULTS_DIR = os.path.join(NEUROFEEDBACK_ROOT, "training_results")

TRAIN_SESSION_PKL = os.path.join(
    NEUROFEEDBACK_TRAINING_RESULTS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"Session_{TRAIN_SESSION_ID}",
    "session_data.pkl",
)

# ─── APP OUTPUT LOCATIONS ────────────────────────────────────────────────
RESULTS_DIR = os.path.join(GRAPH_APP_ROOT, "training_results")
MODELS_DIR  = os.path.join(GRAPH_APP_ROOT, "trained_models")

CURRENT_SESSION_DIR = os.path.join(
    RESULTS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"Session_{CURRENT_SESSION_ID}",
)
os.makedirs(CURRENT_SESSION_DIR, exist_ok=True)

MODEL_OUT_DIR = os.path.join(
    MODELS_DIR,
    f"Subject_{SUBJECT_ID}",
    f"TrainSession_{TRAIN_SESSION_ID}_for_CurrentSession_{CURRENT_SESSION_ID}",
)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ─── GAME PARAMETERS ─────────────────────
NUM_LEVELS        = 1
TRIALS_PER_LEVEL  = 20
INTER_TRIAL_PAUSE = 2.0
INTER_LEVEL_PAUSE = 5.0

CUE_DURATION   = 1.0
TRIAL_DURATION = 10.0

# ─── STREAM + WINDOWING (KEEP 1:1 WITH OTHER PIPELINES) ──────────────────
SAMPLING_RATE     = 256
FEEDBACK_INTERVAL = 0.04
WINDOW_SIZE       = SAMPLING_RATE * 1
STEP_SIZE         = int(FEEDBACK_INTERVAL * SAMPLING_RATE)  # inference cadence

# ─── BASELINE (match game baseline behavior) ─────────────────────────────
BASELINE_SECONDS = 10.0

# ─── PREPROCESSING ───────────────────────────────────────────────────────
APPLY_BANDPASS = True
BP_LO, BP_HI   = 8.0, 30.0
BP_ORDER       = 4

# NOTE: If your offline graph features were built without z-scoring, keep False.
ENABLE_ZSCORE  = False

# ─── PLV + FEATURE SELECTION (TRAIN ONLY) ────────────────────────────────
EPSILON   = 1e-6     # for transform
KL_NBINS  = 20
KL_EPS    = 1e-12
CV_EPS    = 1e-10

# keep low CV (stable) and high KL (discriminative)
CV_KEEP_PCTL = 30
KL_KEEP_PCTL = 70

# Optional smoothing of online predictions
SMOOTH_VOTES = 5
# ─── ONLINE CONTROL (MARGIN + BASELINE CENTERING) ────────────────────────
# Use signed classifier score (margin/logit) instead of hard labels for smoothing.
USE_MARGIN_OUTPUT = True

# Subtract mean baseline margin so "rest" is ~0-mean (kills drift without deadzone/decay).
ENABLE_BASELINE_CENTERING = True

# Optional: ignore first N seconds of baseline margins (filter warm-up) when estimating baseline mean.
BASELINE_WARMUP_SECONDS = 2.0

# ─── LSL ────────────────────────────────────────────────────────────────
LSL_STREAM_TYPE  = "EEG"
LSL_TIMEOUT_SEC  = 5.0
