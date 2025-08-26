# config.py

import os
from glob import glob

# ─── MODE SELECTION ─────────────────────────────────────────────────────
METHOD        = 'plv'      # 'plv' or 'csp'
ADAPTATION    = False       # False = static, True = adaptive
ADAPT_N       = 100        # how many windows to accumulate before adapting

# ─── GAME PARAMETERS ────────────────────────────────────────────────────
NUM_LEVELS        = 10
TRIALS_PER_LEVEL  = 20

# ─── CUE & TRIAL TIMING ─────────────────────────────────────────────────
CUE_DURATION      = 1.0
TRIAL_DURATION    = 6.0

# ─── REAL-TIME FEEDBACK RATE ────────────────────────────────────────────
SAMPLING_RATE      = 256   # Hz
FEEDBACK_INTERVAL  = 0.04  # s
STEP_SIZE          = int(FEEDBACK_INTERVAL * SAMPLING_RATE)

# ─── EEG PROCESSING WINDOW ──────────────────────────────────────────────
WINDOW_SIZE        = 256 * 3   # 3 s @ 256 Hz

# ─── OTHER EEG PARAMETERS ───────────────────────────────────────────────
BUFFER_SIZE        = 10
THRESHOLD          = 3

# ─── CSP CHANNELS ───────────────────────────────────────────────────────
CSP_CHANNELS = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
]

# ─── SUBJECT & SESSION ──────────────────────────────────────────────────
SUBJECT_ID      = "000"
SESSION_ID      = "001"
RESULTS_DIR     = "./collection_results"  # where online session data/finetunes get saved

SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR, f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)

# ─── TRAINED MODEL BASE (offline-trained outputs) ───────────────────────
# This is where train_from_saved.py writes:
#   ...\trained_models\Subject_XXX\Session_001\finetuned_models\csp_lda_static.pkl
#   ...\trained_models\Subject_XXX\Session_001\finetuned_models\gat_finetuned_from_nf.pt
MODELS_BASE = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\neurofeedback\trained_models"

# Legacy/fallback paths (used only if nothing is found above)
CSP_LDA_FALLBACK_PKL   = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\training_data.pkl"
GAT_MODEL_FALLBACK_PT  = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\best_finetuned_model.pt"

# ─── AUTO-RESOLVE TRAINED FILES FOR THIS SUBJECT/SESSION ────────────────
def _resolve_trained_paths(models_base, subject_id, session_id):
    """
    Returns dict with keys:
      'csp_lda_pkl', 'gat_model_pt', 'meta_json', 'root_dir'
    Prefers exact session folder, then latest under the subject, then fallbacks.
    """
    subj_dir = os.path.join(models_base, f"Subject_{subject_id}")
    sess_dir = os.path.join(subj_dir, f"Session_{session_id}", "finetuned_models")
    ret = {
        "csp_lda_pkl": None,
        "gat_model_pt": None,
        "meta_json": None,
        "root_dir": sess_dir
    }

    # 1) Exact session preferred
    if os.path.isdir(sess_dir):
        csp_exact = os.path.join(sess_dir, "csp_lda_static.pkl")
        gat_exact = os.path.join(sess_dir, "gat_finetuned_from_nf.pt")
        meta_exact = os.path.join(sess_dir, "model_meta.json")
        if os.path.isfile(csp_exact):
            ret["csp_lda_pkl"] = csp_exact
        if os.path.isfile(gat_exact):
            ret["gat_model_pt"] = gat_exact
        if os.path.isfile(meta_exact):
            ret["meta_json"] = meta_exact

    # 2) If missing, search latest under the subject
    def _latest(pattern):
        files = glob(pattern, recursive=True)
        if not files:
            return None
        files.sort(key=os.path.getmtime)
        return files[-1]

    if ret["csp_lda_pkl"] is None and os.path.isdir(subj_dir):
        cand = _latest(os.path.join(subj_dir, "Session_*", "finetuned_models", "csp_lda_static.pkl"))
        if cand:
            ret["csp_lda_pkl"] = cand

    if ret["gat_model_pt"] is None and os.path.isdir(subj_dir):
        cand = _latest(os.path.join(subj_dir, "Session_*", "finetuned_models", "gat_finetuned_from_nf.pt"))
        if cand:
            ret["gat_model_pt"] = cand

    if ret["meta_json"] is None and os.path.isdir(subj_dir):
        cand = _latest(os.path.join(subj_dir, "Session_*", "finetuned_models", "model_meta.json"))
        if cand:
            ret["meta_json"] = cand

    # 3) Fallbacks
    if ret["csp_lda_pkl"] is None:
        ret["csp_lda_pkl"] = CSP_LDA_FALLBACK_PKL
    if ret["gat_model_pt"] is None:
        ret["gat_model_pt"] = GAT_MODEL_FALLBACK_PT

    return ret

_RES = _resolve_trained_paths(MODELS_BASE, SUBJECT_ID, SESSION_ID)

# Export the resolved paths for the rest of the codebase:
CSP_LDA_PKL  = _RES["csp_lda_pkl"]
GAT_MODEL_PT = _RES["gat_model_pt"]
MODEL_META   = _RES["meta_json"]  # optional, may be None

# ─── REAL-TIME VISUALISATION ─────────────────────────────────────────────
VISUALISE_PLV   = False
