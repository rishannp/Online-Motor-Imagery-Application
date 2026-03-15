# analyse_sessions.py
#
# Walks all game directories, loads every session_data.pkl, and computes
# a comprehensive set of trial-level and session-level metrics.
#
# Run once from anywhere:  python analyse_sessions.py
# Outputs written to:      ./analysis_outputs/
#
# ── SCREEN GEOMETRY (fixed across all games) ────────────────────────────────
#   CENTER_X      = 400   cursor start / midline
#   LEFT_TARGET   = 30    left paddle contact zone
#   RIGHT_TARGET  = 770   right paddle contact zone
#   LEFT_QUARTER  = 215   midpoint between centre and left target
#   RIGHT_QUARTER = 585   midpoint between centre and right target
#
# ── TRIAL OUTCOME TAXONOMY ──────────────────────────────────────────────────
#
# Every trial falls into exactly one of these outcome categories:
#
#   "hit"                 — cursor contacted the target paddle (original metric)
#   "wrong_paddle"        — cursor contacted the WRONG paddle (strong incorrect signal)
#   "timeout_close_strong"— timed out, final pos crossed quarter-point toward target
#   "timeout_close_weak"  — timed out, final pos on correct side of centre only
#   "timeout_wrong"       — timed out, final pos on wrong side or stuck at centre
#
# These are mutually exclusive and exhaustive.
# "wrong_paddle" is a separate failure mode from timing out — it means the BCI
# actively drove the cursor to the wrong side hard enough to contact that paddle.
#
# ── BROAD ACCURACY ──────────────────────────────────────────────────────────
#   broad_acc = (hit + timeout_close_strong) / n_trials
#   Captures trials where the cursor unambiguously went the right direction.
#
# ── LIBERAL ACCURACY ────────────────────────────────────────────────────────
#   liberal_acc = (hit + timeout_close_strong + timeout_close_weak) / n_trials
#   Captures any trial where the cursor ended on the correct side of centre.
#
# ── FIGHTING FLAG ───────────────────────────────────────────────────────────
#   A trial is "fighting" when:
#     - cursor spent < 45% of frames on the correct side of centre (mostly wrong side)
#     - AND had >= 15 direction reversals (jitter)
#   Interpretation: the BCI was commanding the wrong direction but the user's
#   underlying neural signal was pulling toward the goal, creating visible
#   tug-of-war. Most meaningful for timeout_wrong and wrong_paddle trials.
#
# ── CURSOR METRICS (per trial) ──────────────────────────────────────────────
#   timed_out          bool   ran full trial duration without hitting either paddle
#   final_x            float  last cursor position (pixels)
#   dist_final         float  |final_x - target_x| — end-of-trial distance to goal
#   mean_dist          float  mean(|cursor_x - target_x|) across all frames
#   min_dist           float  closest the cursor ever got to target (best moment)
#   frac_correct_side  float  fraction of frames on correct side of centre [0,1]
#   n_reversals        int    direction sign-changes in velocity (jitter proxy)
#   peak_excursion     float  max displacement from centre in correct direction
#   reaction_frames    int    frames until cursor first crosses centre toward target
#                             (None if cursor never crossed centre in correct direction)
#   trial_duration_s   float  trial duration in seconds (n_frames / 60)

import os
import json
import pickle
import numpy as np
import pandas as pd

# ── SCREEN GEOMETRY ──────────────────────────────────────────────────────────
CENTER_X      = 400.0
LEFT_TARGET   = 60.0   # LEFT_X + PADDLE_W + PLAYER_RADIUS = 20 + 10 + 30
RIGHT_TARGET  = 770.0  # RIGHT_X = SCREEN_W - 20 - PADDLE_W = 770
LEFT_QUARTER  = (CENTER_X + LEFT_TARGET)  / 2.0   # 230.0
RIGHT_QUARTER = (CENTER_X + RIGHT_TARGET) / 2.0   # 585.0
FPS           = 60.0

# A trial is considered timed-out if it has >= this many frames
# (600 = 10s * 60fps; allow 5-frame tolerance for scheduler jitter)
TIMEOUT_FRAME_THRESHOLD = 595

# ── FIGHTING DETECTION THRESHOLDS ────────────────────────────────────────────
# A trial is flagged as "fighting" when:
#   - cursor spent < FIGHTING_CORRECT_SIDE_THRESH of frames on the correct side
#     (i.e. predominantly on the wrong side)
#   - AND reversals on the WRONG side >= FIGHTING_REVERSAL_THRESH
#
# Reversals are split by which side of centre the cursor is on at each step:
#   n_reversals_wrong  — direction changes while on the wrong side (fighting signal)
#   n_reversals_correct— direction changes while on the correct side (near-target noise,
#                        NOT used in the fighting flag — high reversals on the correct
#                        side is evidence of control, not failure)
FIGHTING_CORRECT_SIDE_THRESH = 0.45
FIGHTING_REVERSAL_THRESH     = 15

# ── GAME DIRECTORIES ─────────────────────────────────────────────────────────
EXPERIMENT_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts"

GAME_DIRS = {
    "csp":                 os.path.join(EXPERIMENT_ROOT, "csp",                 "training_results"),
    "graph_ml":            os.path.join(EXPERIMENT_ROOT, "graph_ml",            "training_results"),
    "graph_neurofeedback": os.path.join(EXPERIMENT_ROOT, "graph_neurofeedback", "training_results"),
    "neurofeedback":       os.path.join(EXPERIMENT_ROOT, "neurofeedback",       "training_results"),
}

OUTPUT_DIR = "./analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def target_x(label: int) -> float:
    return LEFT_TARGET if label == 0 else RIGHT_TARGET


def classify_outcome(cursor_x: np.ndarray, label: int, hit: bool) -> str:
    """
    Assign one of five mutually exclusive outcome categories to a trial.
    Requires knowing whether a hit occurred (from the pkl 'hit' field).
    For wrong_paddle: a non-hit trial that ended at the opposite paddle zone.
    """
    if hit:
        return "hit"

    final = float(cursor_x[-1])

    # Wrong paddle: cursor ended deep on the wrong side (past the wrong quarter-point)
    if label == 0 and final > RIGHT_QUARTER:
        return "wrong_paddle"
    if label == 1 and final < LEFT_QUARTER:
        return "wrong_paddle"

    timed_out = len(cursor_x) >= TIMEOUT_FRAME_THRESHOLD

    if label == 0:
        if final < LEFT_QUARTER:
            return "timeout_close_strong" if timed_out else "close_strong_early"
        if final < CENTER_X:
            return "timeout_close_weak"   if timed_out else "close_weak_early"
        return "timeout_wrong" if timed_out else "wrong_early"
    else:
        if final > RIGHT_QUARTER:
            return "timeout_close_strong" if timed_out else "close_strong_early"
        if final > CENTER_X:
            return "timeout_close_weak"   if timed_out else "close_weak_early"
        return "timeout_wrong" if timed_out else "wrong_early"


def split_reversals(cursor_x: np.ndarray, label: int):
    """
    Count direction reversals separately for frames spent on the wrong side
    and frames spent on the correct side of centre.

    A reversal at frame i is assigned to whichever side the cursor is on at
    frame i (the frame where the direction change occurs).

    Returns (n_reversals_wrong, n_reversals_correct).
    """
    vel   = np.diff(cursor_x)                          # [N-1]
    signs = np.sign(vel)                               # -1, 0, +1

    # Reversal occurs where sign changes between consecutive non-zero steps
    # Use the position at the frame where the reversal is detected (index i+1
    # in cursor_x, corresponding to index i in vel)
    rev_indices = np.where(np.diff(signs) != 0)[0]    # indices into vel

    n_wrong   = 0
    n_correct = 0
    for i in rev_indices:
        pos = cursor_x[i + 1]                         # position at reversal frame
        if label == 0:
            on_correct = pos < CENTER_X
        else:
            on_correct = pos > CENTER_X
        if on_correct:
            n_correct += 1
        else:
            n_wrong += 1

    return int(n_wrong), int(n_correct)


def fighting_flag(cursor_x: np.ndarray, label: int) -> bool:
    """
    True when the cursor was predominantly on the wrong side of centre
    AND showed high jitter specifically on the wrong side.

    Uses wrong-side reversals only — oscillations on the correct side
    are evidence of control attempts near the target, not fighting.
    """
    frac_correct = float(
        (cursor_x < CENTER_X).mean() if label == 0 else (cursor_x > CENTER_X).mean()
    )
    n_rev_wrong, _ = split_reversals(cursor_x, label)
    return (frac_correct < FIGHTING_CORRECT_SIDE_THRESH) and (n_rev_wrong >= FIGHTING_REVERSAL_THRESH)


def reaction_frames(cursor_x: np.ndarray, label: int):
    """
    Number of frames until cursor first crosses centre in the correct direction.
    Returns None if it never does.
    """
    if label == 0:
        crossed = np.where(cursor_x < CENTER_X)[0]
    else:
        crossed = np.where(cursor_x > CENTER_X)[0]
    return int(crossed[0]) if len(crossed) > 0 else None


def trial_metrics(tid: int, trial: dict) -> dict:
    """Compute all metrics for a single trial."""
    cx    = np.asarray(trial["cursor_x"], dtype=float)
    label = int(trial["label"])
    hit   = bool(trial["hit"])
    tgt   = target_x(label)

    n         = len(cx)
    timed_out = n >= TIMEOUT_FRAME_THRESHOLD
    outcome   = classify_outcome(cx, label, hit)
    fighting  = fighting_flag(cx, label)

    vel       = np.diff(cx)
    reversals = int(np.sum(np.diff(np.sign(vel)) != 0))   # total (kept for reference)
    n_rev_wrong, n_rev_correct = split_reversals(cx, label)

    frac_correct = float(
        (cx < CENTER_X).mean() if label == 0 else (cx > CENTER_X).mean()
    )

    if label == 0:
        peak_excursion = float(max(0.0, CENTER_X - cx.min()))
    else:
        peak_excursion = float(max(0.0, cx.max() - CENTER_X))

    dist = np.abs(cx - tgt)
    rxn  = reaction_frames(cx, label)

    return {
        "trial_id":          tid,
        "label":             label,
        "label_name":        "LEFT" if label == 0 else "RIGHT",
        # outcome
        "hit":               hit,
        "timed_out":         timed_out,
        "outcome":           outcome,
        "fighting":          fighting,
        # position at end of trial
        "final_x":           float(cx[-1]),
        "dist_final":        float(dist[-1]),
        # trajectory summary
        "mean_dist":         float(dist.mean()),
        "min_dist":          float(dist.min()),
        "frac_correct_side": frac_correct,
        "n_reversals":       reversals,
        "n_reversals_wrong": n_rev_wrong,
        "n_reversals_correct": n_rev_correct,
        "peak_excursion":    peak_excursion,
        # timing
        "n_frames":          n,
        "trial_duration_s":  float(n / FPS),
        "reaction_frames":   rxn,
        "reaction_s":        float(rxn / FPS) if rxn is not None else None,
    }


def session_summary(trial_rows: list, game: str, subject: str,
                    session: str, config: dict) -> dict:
    """Aggregate trial-level metrics into one session-level row."""
    df = pd.DataFrame(trial_rows)
    n  = len(df)

    left_df  = df[df["label"] == 0]
    right_df = df[df["label"] == 1]

    # Outcome counts — exhaustive
    oc = df["outcome"].value_counts().to_dict()
    def oc_get(k): return int(oc.get(k, 0))

    n_hit                 = oc_get("hit")
    n_wrong_paddle        = oc_get("wrong_paddle")
    n_to_close_strong     = oc_get("timeout_close_strong")
    n_to_close_weak       = oc_get("timeout_close_weak")
    n_to_wrong            = oc_get("timeout_wrong")
    n_close_strong_early  = oc_get("close_strong_early")
    n_close_weak_early    = oc_get("close_weak_early")
    n_wrong_early         = oc_get("wrong_early")

    # Timeout totals
    n_timed_out      = int(df["timed_out"].sum())
    n_timed_out_good = n_to_close_strong   # timed out but on correct side past quarter-point
    n_timed_out_ok   = n_to_close_weak     # timed out on correct side, not past quarter-point
    n_timed_out_bad  = n_to_wrong          # timed out on wrong side / stuck

    # Accuracy tiers
    hit_rate     = n_hit / n if n > 0 else 0.0
    broad_acc    = (n_hit + n_to_close_strong) / n if n > 0 else 0.0
    liberal_acc  = (n_hit + n_to_close_strong + n_to_close_weak) / n if n > 0 else 0.0

    # Fighting breakdown
    fighting_df          = df[df["fighting"]]
    n_fighting           = len(fighting_df)
    n_fighting_timeout   = int((fighting_df["timed_out"]).sum())
    n_fighting_wrong_paddle = int((fighting_df["outcome"] == "wrong_paddle").sum())

    # Reaction time (only trials where cursor crossed centre at all)
    rxn_valid = df["reaction_s"].dropna()

    # Per-class accuracy
    def class_broad(sub_df):
        if len(sub_df) == 0: return float("nan")
        return sub_df["outcome"].isin(["hit","timeout_close_strong"]).mean()

    def class_liberal(sub_df):
        if len(sub_df) == 0: return float("nan")
        return sub_df["outcome"].isin(["hit","timeout_close_strong","timeout_close_weak"]).mean()

    return {
        # identifiers
        "game":                      game,
        "subject":                   subject,
        "session":                   session,
        "plv_method":                (config.get("PLV_METHOD","N/A").strip("'\"")
                                      if game == "graph_neurofeedback" else "N/A"),
        # counts
        "n_trials":                  n,
        "n_left_trials":             len(left_df),
        "n_right_trials":            len(right_df),
        # ── outcome counts (exhaustive) ──────────────────────────────────
        "n_hit":                     n_hit,
        "n_wrong_paddle":            n_wrong_paddle,
        "n_timeout_close_strong":    n_to_close_strong,
        "n_timeout_close_weak":      n_to_close_weak,
        "n_timeout_wrong":           n_to_wrong,
        "n_close_strong_early":      n_close_strong_early,
        "n_close_weak_early":        n_close_weak_early,
        "n_wrong_early":             n_wrong_early,
        # ── timeout summary ──────────────────────────────────────────────
        "n_timed_out":               n_timed_out,
        "n_timed_out_close_strong":  n_timed_out_good,
        "n_timed_out_close_weak":    n_timed_out_ok,
        "n_timed_out_wrong":         n_timed_out_bad,
        # ── accuracy metrics ─────────────────────────────────────────────
        "hit_rate":                  hit_rate,
        "broad_acc":                 broad_acc,
        "liberal_acc":               liberal_acc,
        "hit_rate_left":             left_df["hit"].mean()  if len(left_df)  > 0 else float("nan"),
        "hit_rate_right":            right_df["hit"].mean() if len(right_df) > 0 else float("nan"),
        "broad_acc_left":            class_broad(left_df),
        "broad_acc_right":           class_broad(right_df),
        "liberal_acc_left":          class_liberal(left_df),
        "liberal_acc_right":         class_liberal(right_df),
        # ── cursor quality (session means) ───────────────────────────────
        "mean_dist_final":           df["dist_final"].mean(),
        "mean_dist_trajectory":      df["mean_dist"].mean(),
        "mean_min_dist":             df["min_dist"].mean(),
        "mean_frac_correct_side":    df["frac_correct_side"].mean(),
        "mean_reversals":            df["n_reversals"].mean(),
        "mean_reversals_wrong":      df["n_reversals_wrong"].mean(),
        "mean_reversals_correct":    df["n_reversals_correct"].mean(),
        "mean_peak_excursion":       df["peak_excursion"].mean(),
        # ── per-class cursor quality ──────────────────────────────────────
        "mean_dist_final_left":      left_df["dist_final"].mean()          if len(left_df)  > 0 else float("nan"),
        "mean_dist_final_right":     right_df["dist_final"].mean()         if len(right_df) > 0 else float("nan"),
        "mean_frac_correct_left":    left_df["frac_correct_side"].mean()   if len(left_df)  > 0 else float("nan"),
        "mean_frac_correct_right":   right_df["frac_correct_side"].mean()  if len(right_df) > 0 else float("nan"),
        # ── reaction time ─────────────────────────────────────────────────
        "mean_reaction_s":           float(rxn_valid.mean()) if len(rxn_valid) > 0 else float("nan"),
        "median_reaction_s":         float(rxn_valid.median()) if len(rxn_valid) > 0 else float("nan"),
        "n_no_reaction":             int(df["reaction_s"].isna().sum()),
        # ── trial duration ────────────────────────────────────────────────
        "mean_trial_duration_s":     df["trial_duration_s"].mean(),
        "mean_hit_duration_s":       (df[df["hit"]]["trial_duration_s"].mean()
                                      if df["hit"].any() else float("nan")),
        # ── fighting ─────────────────────────────────────────────────────
        "n_fighting":                n_fighting,
        "frac_fighting":             n_fighting / n if n > 0 else 0.0,
        "n_fighting_timeout":        n_fighting_timeout,
        "n_fighting_wrong_paddle":   n_fighting_wrong_paddle,
    }


# ── DIRECTORY WALKING ─────────────────────────────────────────────────────────

def load_config(session_dir: str) -> dict:
    cfg_path = os.path.join(session_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def walk_game(game_name: str, results_dir: str):
    """Yield (subject_id, session_id, session_dir, config, trials) for every valid session."""
    if not os.path.isdir(results_dir):
        print(f"  [WARN] not found: {results_dir}")
        return

    for subj in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not subj.is_dir():
            continue
        for sess in sorted(os.scandir(subj.path), key=lambda e: e.name):
            if not sess.is_dir():
                continue
            pkl_path = os.path.join(sess.path, "session_data.pkl")
            if not os.path.isfile(pkl_path):
                continue
            try:
                with open(pkl_path, "rb") as f:
                    trials = pickle.load(f)
            except Exception as e:
                print(f"  [WARN] could not load {pkl_path}: {e}")
                continue
            yield subj.name, sess.name, sess.path, load_config(sess.path), trials


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    all_trial_rows   = []
    all_session_rows = []

    for game_name, results_dir in GAME_DIRS.items():
        print(f"\n{'='*70}")
        print(f"Game: {game_name}")
        print(f"  Dir: {results_dir}")

        game_trial_rows   = []
        game_session_rows = []

        for subject_id, session_id, session_dir, cfg, trials in walk_game(game_name, results_dir):
            print(f"  {subject_id} / {session_id}  ({len(trials)} trials)")

            trial_rows = []
            for tid, trial in trials.items():
                cx = trial.get("cursor_x", None)
                if cx is None or len(cx) == 0 or "label" not in trial:
                    continue
                row = trial_metrics(tid, trial)
                row["game"]    = game_name
                row["subject"] = subject_id
                row["session"] = session_id
                trial_rows.append(row)

            if not trial_rows:
                continue

            sess_row = session_summary(trial_rows, game_name, subject_id, session_id, cfg)
            game_trial_rows.extend(trial_rows)
            game_session_rows.append(sess_row)

        if not game_trial_rows:
            continue

        trials_df   = pd.DataFrame(game_trial_rows)
        sessions_df = pd.DataFrame(game_session_rows)

        trials_path   = os.path.join(OUTPUT_DIR, f"{game_name}_trials.csv")
        sessions_path = os.path.join(OUTPUT_DIR, f"{game_name}_sessions.csv")
        trials_df.to_csv(trials_path,   index=False)
        sessions_df.to_csv(sessions_path, index=False)

        all_trial_rows.extend(game_trial_rows)
        all_session_rows.extend(game_session_rows)

        # Console summary
        print(f"\n  {game_name} — session summary:")
        cols = ["subject","session","n_trials","hit_rate","broad_acc","liberal_acc",
                "n_timed_out","n_timed_out_close_strong","n_timed_out_wrong",
                "n_wrong_paddle","n_fighting","mean_dist_final","mean_reaction_s"]
        print(sessions_df[cols].to_string(index=False))

    if not all_trial_rows:
        print("\nNo data found. Check EXPERIMENT_ROOT and directory names.")
        return

    all_trials_df   = pd.DataFrame(all_trial_rows)
    all_sessions_df = pd.DataFrame(all_session_rows)
    all_trials_df.to_csv(  os.path.join(OUTPUT_DIR, "all_trials.csv"),   index=False)
    all_sessions_df.to_csv(os.path.join(OUTPUT_DIR, "all_sessions.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"COMBINED — {len(all_sessions_df)} sessions, "
          f"{len(all_trials_df)} trials across {sum(1 for v in GAME_DIRS.values() if os.path.isdir(v))} games found")
    print(all_sessions_df[[
        "game","subject","session",
        "hit_rate","broad_acc","liberal_acc",
        "n_hit","n_timeout_close_strong","n_timeout_close_weak","n_timeout_wrong",
        "n_wrong_paddle","n_fighting",
        "mean_dist_final","mean_min_dist","mean_reaction_s",
    ]].to_string(index=False))

    print(f"\nOutputs written to: {os.path.abspath(OUTPUT_DIR)}/")
    print("  all_trials.csv              — one row per trial, all games")
    print("  all_sessions.csv            — one row per session, all games")
    print("  <game>_trials.csv           — per-game trial-level data")
    print("  <game>_sessions.csv         — per-game session-level data")


if __name__ == "__main__":
    main()