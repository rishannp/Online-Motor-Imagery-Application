# chance_level_simulation.py
#
# Computes empirical chance levels for the BCI cursor task via Monte Carlo
# simulation of a symmetric random walk, using step-size parameters drawn
# from actual session data.
#
# Key finding: this is NOT a 50% chance task.
# The paddle boundaries are ~370px from centre. A cursor performing a
# symmetric random walk (pure noise BCI) almost never reaches the paddle
# in 10 seconds, so:
#
#   Chance hit_rate    ≈ 0%    (paddle contact essentially impossible by noise)
#   Chance broad_acc   ≈ 0%    (quarter-point threshold rarely crossed by noise)
#   Chance liberal_acc ≈ 49%   (cursor ends on correct side ~50% by symmetry)
#
# The appropriate chance line therefore depends on which metric you are plotting:
#   hit_rate / broad_acc  → chance ≈ 0%   (any above-zero performance is real)
#   liberal_acc           → chance ≈ 50%  (standard binomial interpretation holds)
#   frac_correct_side     → chance ≈ 50%  (symmetric by construction)
#
# HOW IT WORKS:
#   1. Load all session_data.pkl files from the experiment tree.
#   2. For each session, extract the empirical step distribution from cursor_x
#      (steps are always -2, 0, or +2 pixels — the BCI outputs a ternary command
#      multiplied by a fixed step size).
#   3. Symmetrise the move probabilities to create a null BCI (equal P(left)=P(right)).
#      The hold probability P(0) is kept as-is since it reflects the dead-zone
#      setting, not the BCI's directional accuracy.
#   4. Simulate N trials per session under this null BCI and compute the full
#      metric suite (hit, broad, liberal, frac_correct_side, mean_dist_final).
#   5. Report per-session chance distributions with 95% CI via bootstrap.
#
# OUTPUT:
#   chance_levels.csv  — per-session chance estimates + CI
#   chance_levels.png  — visualisation overlaid on actual session metrics
#
# USAGE (Spyder):
#   Run as-is, or adjust EXPERIMENT_ROOT and N_SIMULATIONS at the top.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

# ── configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_ROOT = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts"

GAME_DIRS = {
    "csp":                 os.path.join(EXPERIMENT_ROOT, "csp",                 "training_results"),
    "graph_ml":            os.path.join(EXPERIMENT_ROOT, "graph_ml",            "training_results"),
    "graph_neurofeedback": os.path.join(EXPERIMENT_ROOT, "graph_neurofeedback", "training_results"),
    "neurofeedback":       os.path.join(EXPERIMENT_ROOT, "neurofeedback",       "training_results"),
}

OUTPUT_DIR      = "./simulation_outputs"
N_SIMULATIONS   = 5_000   # per session; increase to 50k for tighter CIs (slower)
N_BOOTSTRAP     = 1_000    # bootstrap resamples for CI estimation
RANDOM_SEED     = 42
CI_LEVEL        = 0.95     # confidence interval width

# ── screen geometry (fixed across all games) ──────────────────────────────────
CENTER_X      = 400.0
LEFT_PAD      = 60.0
RIGHT_PAD     = 770.0
LEFT_QUARTER  = (CENTER_X + LEFT_PAD)  / 2.0   # 215.0
RIGHT_QUARTER = (CENTER_X + RIGHT_PAD) / 2.0   # 585.0
STEP_SIZE     = 2.0
MAX_FRAMES    = 598   # 10s at 60fps with tolerance
FPS           = 60.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def natural_key(s):
    import re
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', s)]


def extract_step_probs(trials: dict):
    """
    Pull all cursor steps from a session and return empirical probabilities.
    Steps are always -2, 0, or +2. Returns (p_pos, p_neg, p_zero).
    """
    all_steps = []
    for trial in trials.values():
        cx = np.asarray(trial.get('cursor_x', []), dtype=float)
        if len(cx) > 1:
            all_steps.extend(np.diff(cx).tolist())

    if not all_steps:
        return 0.465, 0.465, 0.070  # sensible default

    steps = np.array(all_steps)
    p_pos  = float((steps > 0).mean())
    p_neg  = float((steps < 0).mean())
    p_zero = float((steps == 0).mean())
    return p_pos, p_neg, p_zero


def classify_outcome(x_final, label, hit):
    """Same taxonomy as analyse_sessions.py."""
    if hit:
        return 'hit'
    if label == 0 and x_final > RIGHT_QUARTER:
        return 'wrong_paddle'
    if label == 1 and x_final < LEFT_QUARTER:
        return 'wrong_paddle'
    if label == 0:
        if x_final < LEFT_QUARTER:  return 'timeout_close_strong'
        if x_final < CENTER_X:      return 'timeout_close_weak'
        return 'timeout_wrong'
    else:
        if x_final > RIGHT_QUARTER: return 'timeout_close_strong'
        if x_final > CENTER_X:      return 'timeout_close_weak'
        return 'timeout_wrong'


def simulate_session(n_trials, p_pos, p_neg, p_zero,
                     trial_lengths, rng):
    """
    Simulate n_trials under a SYMMETRISED (chance) BCI.

    p_move = (p_pos + p_neg) / 2  — equal probability each direction
    p_zero preserved from data    — dead-zone/hold behaviour unchanged

    trial_lengths: array of actual frame counts from real trials,
                   used so simulated trials have realistic durations.

    Returns dict of metric arrays, one value per simulated trial.
    """
    p_move = (p_pos + p_neg) / 2.0   # symmetrise

    outcomes          = []
    frac_correct      = []
    dist_finals       = []
    min_dists         = []
    mean_dists        = []

    for i in range(n_trials):
        label      = i % 2                                    # alternate L/R
        max_frames = int(trial_lengths[i % len(trial_lengths)])
        tgt        = LEFT_PAD if label == 0 else RIGHT_PAD

        x        = CENTER_X
        hit      = False
        x_hist   = [x]

        for _ in range(max_frames):
            r = rng.random()
            if r < p_move:
                x += STEP_SIZE
            elif r < 2 * p_move:
                x -= STEP_SIZE
            # else: hold

            x_hist.append(x)

            if x >= RIGHT_PAD:
                x    = RIGHT_PAD
                hit  = (label == 1)
                break
            if x <= LEFT_PAD:
                x    = LEFT_PAD
                hit  = (label == 0)
                break

        x_arr = np.asarray(x_hist, dtype=float)
        dist  = np.abs(x_arr - tgt)

        outcome = classify_outcome(x, label, hit)
        outcomes.append(outcome)

        if label == 0:
            frac_correct.append(float((x_arr < CENTER_X).mean()))
        else:
            frac_correct.append(float((x_arr > CENTER_X).mean()))

        dist_finals.append(float(dist[-1]))
        min_dists.append(float(dist.min()))
        mean_dists.append(float(dist.mean()))

    outcomes = np.array(outcomes)
    n = len(outcomes)

    hit_arr    = outcomes == 'hit'
    strong_arr = outcomes == 'timeout_close_strong'
    weak_arr   = outcomes == 'timeout_close_weak'

    return {
        'hit_rate':          hit_arr.mean(),
        'broad_acc':         (hit_arr | strong_arr).mean(),
        'liberal_acc':       (hit_arr | strong_arr | weak_arr).mean(),
        'frac_correct_side': np.mean(frac_correct),
        'mean_dist_final':   np.mean(dist_finals),
        'mean_min_dist':     np.mean(min_dists),
        'mean_dist_traj':    np.mean(mean_dists),
        'n_hit':             hit_arr.sum(),
        'n_strong':          strong_arr.sum(),
        'n_weak':            weak_arr.sum(),
        'n_wrong':           (outcomes == 'timeout_wrong').sum(),
        'n_wrong_paddle':    (outcomes == 'wrong_paddle').sum(),
    }


def bootstrap_ci(values, n_boot, ci, rng):
    """Return (mean, lower, upper) via percentile bootstrap."""
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (float(np.mean(values)),
            float(np.percentile(boots, 100 * alpha)),
            float(np.percentile(boots, 100 * (1 - alpha))))


# ── session walker ────────────────────────────────────────────────────────────

def walk_sessions():
    for game, results_dir in GAME_DIRS.items():
        if not os.path.isdir(results_dir):
            print(f"  [WARN] not found: {results_dir}")
            continue
        for subj_entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
            if not subj_entry.is_dir():
                continue
            for sess_entry in sorted(os.scandir(subj_entry.path), key=lambda e: e.name):
                if not sess_entry.is_dir():
                    continue
                pkl = os.path.join(sess_entry.path, 'session_data.pkl')
                if not os.path.isfile(pkl):
                    continue
                try:
                    with open(pkl, 'rb') as f:
                        trials = pickle.load(f)
                except Exception as e:
                    print(f"  [WARN] {pkl}: {e}")
                    continue
                yield game, subj_entry.name, sess_entry.name, trials


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    METRICS = ['hit_rate', 'broad_acc', 'liberal_acc',
               'frac_correct_side', 'mean_dist_final', 'mean_min_dist']

    rows = []

    for game, subject, session, trials in walk_sessions():
        n_trials = len(trials)
        print(f"  {game} / {subject} / {session}  ({n_trials} trials)")

        p_pos, p_neg, p_zero = extract_step_probs(trials)
        p_move = (p_pos + p_neg) / 2.0

        # Actual trial frame counts for realistic duration sampling
        trial_lengths = np.array([
            len(t['cursor_x']) if len(t.get('cursor_x', [])) > 0 else MAX_FRAMES
            for t in trials.values()
        ], dtype=int)
        trial_lengths = np.clip(trial_lengths, 1, MAX_FRAMES)

        # Run N_SIMULATIONS independent single-session simulations
        # Each simulation produces one estimate of each metric
        sim_results = defaultdict(list)
        for _ in range(N_SIMULATIONS):
            res = simulate_session(n_trials, p_pos, p_neg, p_zero,
                                   trial_lengths, rng)
            for k, v in res.items():
                sim_results[k].append(v)

        # Bootstrap CI on the distribution of simulation estimates
        row = {
            'game':    game,
            'subject': subject,
            'session': session,
            'n_trials': n_trials,
            'p_move_empirical': p_pos + p_neg,    # total move prob in real data
            'p_move_chance':    2 * p_move,        # symmetrised (same total)
            'p_hold':           p_zero,
        }

        for m in METRICS:
            vals = np.array(sim_results[m])
            mean, lo, hi = bootstrap_ci(vals, N_BOOTSTRAP, CI_LEVEL, rng)
            row[f'chance_{m}']    = mean
            row[f'chance_{m}_lo'] = lo
            row[f'chance_{m}_hi'] = hi

        rows.append(row)

    if not rows:
        print("No sessions found. Check EXPERIMENT_ROOT.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, 'chance_levels.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CHANCE LEVEL SUMMARY (mean across all sessions)")
    print(f"  N_SIMULATIONS per session: {N_SIMULATIONS:,}")
    print(f"  CI level: {int(CI_LEVEL*100)}%")
    print()

    for m in ['hit_rate', 'broad_acc', 'liberal_acc', 'frac_correct_side']:
        vals = df[f'chance_{m}']
        print(f"  {m:22s}  mean={vals.mean():.4f}  "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    print()
    print("Interpretation:")
    print("  hit_rate / broad_acc  → chance ≈ 0%  (paddles unreachable by noise)")
    print("  liberal_acc           → chance ≈ 50% (cursor ends on correct side ~half the time)")
    print("  frac_correct_side     → chance ≈ 50% (symmetric random walk)")

    # ── load actual session metrics for comparison if available ───────────────
    actual_csv = os.path.join(OUTPUT_DIR, 'all_sessions.csv')
    if os.path.isfile(actual_csv):
        _plot_comparison(df, actual_csv)
    else:
        print("\nall_sessions.csv not found — skipping comparison plot.")
        print("Run analyse_sessions.py first if you want the overlay plot.")
        _plot_chance_only(df)


def _plot_chance_only(df):
    """Plot chance distribution when actual data is unavailable."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#0d0f14')

    metrics = [
        ('chance_hit_rate',    'Hit Rate',    '#4ade80'),
        ('chance_broad_acc',   'Broad Acc',   '#60a5fa'),
        ('chance_liberal_acc', 'Liberal Acc', '#a78bfa'),
    ]

    for ax, (col, label, color) in zip(axes, metrics):
        ax.set_facecolor('#141720')
        vals = df[col].values
        ax.hist(vals, bins=20, color=color, alpha=0.7, edgecolor='none')
        ax.axvline(vals.mean(), color='white', lw=1.5, linestyle='--', label=f'mean={vals.mean():.3f}')
        ax.set_title(label, color='white', fontsize=11)
        ax.set_xlabel('Chance probability', color='#8892a4')
        ax.tick_params(colors='#8892a4')
        for spine in ax.spines.values():
            spine.set_edgecolor('#252a38')
        ax.legend(fontsize=8, labelcolor='white', facecolor='#1e2435')

    fig.suptitle('Chance Level Distributions (Monte Carlo)', color='white', fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'chance_levels.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.show()


def _plot_comparison(chance_df, actual_csv):
    """
    Overlay chance CIs on actual session performance.
    One panel per metric, one dot per session, horizontal line = chance mean.
    """
    actual = pd.read_csv(actual_csv)

    # Merge on game + subject + session
    merged = actual.merge(
        chance_df[['game','subject','session',
                   'chance_hit_rate','chance_hit_rate_lo','chance_hit_rate_hi',
                   'chance_broad_acc','chance_broad_acc_lo','chance_broad_acc_hi',
                   'chance_liberal_acc','chance_liberal_acc_lo','chance_liberal_acc_hi']],
        on=['game','subject','session'], how='left'
    )

    GAME_COLOR = {
        'neurofeedback':       '#06b6d4',
        'graph_neurofeedback': '#ec4899',
        'csp':                 '#f59e0b',
        'graph_ml':            '#f97316',
    }

    metrics = [
        ('hit_rate',    'chance_hit_rate',    'Hit Rate'),
        ('broad_acc',   'chance_broad_acc',   'Broad Accuracy'),
        ('liberal_acc', 'chance_liberal_acc', 'Liberal Accuracy'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor('#0d0f14')

    for ax, (actual_col, chance_col, title) in zip(axes, metrics):
        ax.set_facecolor('#141720')
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title, color='white', fontsize=11, pad=10)
        ax.set_xlabel('Session (index)', color='#8892a4', fontsize=9)
        ax.set_ylabel('Accuracy', color='#8892a4', fontsize=9)
        ax.tick_params(colors='#8892a4')
        for spine in ax.spines.values():
            spine.set_edgecolor('#252a38')
        ax.axhline(0.5, color='#3a4060', lw=1, linestyle='--', alpha=0.6)
        ax.text(0, 0.51, '0.50', color='#3a4060', fontsize=7)

        for i, row in merged.iterrows():
            color = GAME_COLOR.get(row['game'], '#888')
            # actual value
            ax.scatter(i, row[actual_col], color=color, s=30, zorder=4, alpha=0.9)
            # chance CI as shaded range
            if not pd.isna(row.get(chance_col + '_lo', np.nan)):
                ax.fill_between(
                    [i - 0.3, i + 0.3],
                    [row[chance_col + '_lo']] * 2,
                    [row[chance_col + '_hi']] * 2,
                    color='white', alpha=0.08, zorder=2
                )
                ax.scatter(i, row[chance_col], color='white', s=8,
                           marker='_', zorder=3, alpha=0.5)

    # Legend
    legend_els = [
        mpatches.Patch(color=c, label=g)
        for g, c in GAME_COLOR.items()
    ] + [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='w',
               markersize=5, label='Actual', linewidth=0),
        Line2D([0],[0], marker='_', color='w', markersize=8,
               label='Chance (95% CI)', linewidth=0),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=6,
               facecolor='#1e2435', labelcolor='white', fontsize=8,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Actual Performance vs Chance Level', color='white', fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUTPUT_DIR, 'chance_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.show()


if __name__ == '__main__':
    main()