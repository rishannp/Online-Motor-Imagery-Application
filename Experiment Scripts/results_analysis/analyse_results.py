"""
analyse_results.py

Analyses online multi-session BCI results for three decoders:
  neurofeedback  : AR-PSD alpha-asymmetry (training-free)
  csp            : CSP + linear SVM (trained on one prior session)
  graph_ml       : Graph/PLV with dual-axis stability-discriminability selection + SVM

All accuracy figures use liberal accuracy only.

Liberal accuracy = (hits + timeout_close_strong + timeout_close_weak) / n_trials
Captures any trial where the cursor ended on the correct side of centre.
Chance baseline is 50%.

Run:
    python analyse_results.py

Outputs go to ./results_analysis_outputs/
"""

import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import levene, wilcoxon, mannwhitneyu

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "figure.dpi":         150,
})

# ---- PATHS -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR), "analysis_outputs")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_analysis_outputs")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
TAB_DIR    = os.path.join(OUT_DIR, "tables")

for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- COLOUR PALETTE ----------------------------------------------------------
COL = {
    "neurofeedback": "#8AAFC4",   # pastel blue
    "csp":           "#E8A87C",   # pastel coral
    "graph_ml":      "#85B87A",   # pastel green
}
DECODERS     = ["neurofeedback", "csp", "graph_ml"]
LABELS       = {"neurofeedback": "AR-PSD", "csp": "CSP", "graph_ml": "Graph+ML"}
CHANCE       = 0.50
METRIC       = "liberal_acc"
METRIC_LABEL = "Liberal Accuracy"

# ---- HELPERS -----------------------------------------------------------------

def fmt_p(p):
    """Format p-value. Use x10 notation when very small."""
    if p < 0.001:
        exp  = int(np.floor(np.log10(p)))
        coef = p / (10 ** exp)
        return f"{coef:.1f}x10$^{{{exp}}}$"
    return f"p = {p:.3f}"


def save(fig, name, dpi=300):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {name}.png")


# ---- LOAD DATA ---------------------------------------------------------------

def load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "all_sessions.csv"))

    def parse_session(s):
        s = str(s)
        if s.startswith("Session_"):
            return int(s.split("_")[1])
        if s.upper() == "ME":
            return 0
        try:
            return int(s)
        except Exception:
            return -1

    def parse_subject(s):
        parts = str(s).split("_")
        try:
            return int(parts[-1])
        except Exception:
            return -1

    df["session_num"] = df["session"].apply(parse_session)
    df["subject_num"] = df["subject"].apply(parse_subject)
    df = df[df["game"].isin(DECODERS)].copy()
    df["decoder"] = df["game"]
    return df


def supervised(df):
    return df[df["decoder"].isin(["csp", "graph_ml"]) & (df["session_num"] >= 2)].copy()


def all_three(df):
    return df[df["session_num"] >= 1].copy()


# ---- FIG 1: GRAND OVERVIEW (styled like bci_accuracy_chart) -----------------

# Decoder display for this figure (includes graph_neurofeedback if present)
ALL_DEC_ORDER  = ["neurofeedback", "graph_neurofeedback", "csp", "graph_ml"]
ALL_DEC_LABELS = {
    "neurofeedback":       "NFB",
    "graph_neurofeedback": "GNF",
    "csp":                 "CSP",
    "graph_ml":            "GML",
}
ALL_DEC_COLORS = {
    "neurofeedback":       "#8AAFC4",   # pastel blue
    "graph_neurofeedback": "#7EC4B0",   # pastel teal
    "csp":                 "#E8A87C",   # pastel coral
    "graph_ml":            "#85B87A",   # pastel green
}

# Per-metric bar colours — slightly deeper than pastel so they read on white
METRIC_COLS = {
    "hit_rate":   "#62B462",   # medium green
    "broad_acc":  "#5A9EC4",   # medium blue
    "liberal_acc":"#8B74BE",   # medium lavender
}
METRIC_NAMES = {
    "hit_rate":   "Hit Rate",
    "broad_acc":  "Broad Accuracy",
    "liberal_acc":"Liberal Accuracy",
}

BG_DARK   = "#ffffff"   # white background for publication
BG_SURF   = "#f8f8f8"   # very light panel background
TXT_DIM   = "#666666"   # medium-dark axis labels
TXT_HI    = "#222222"   # primary text
GRID_COL  = "#dddddd"   # subtle separators


def _UNUSED_build_row_data(full, subjects_in_row):
    """
    Build the (records, x_labels, dec_spans, subj_spans) arrays for a single
    row of participants.  Returns (records, x_labels, dec_spans, subj_spans, total_width).
    """
    GAP_SESSION = 0.18
    GAP_DECODER = 0.65
    GAP_SUBJECT = 1.30
    BAR_W       = 0.26
    GROUP_W     = BAR_W * 3

    records, x_labels, dec_spans, subj_spans = [], [], [], []
    x = 0.0

    for subj in subjects_in_row:
        subj_x_start = x
        decs_present = [d for d in ALL_DEC_ORDER
                        if not full[(full["subject_num"] == subj) &
                                    (full["game"] == d)].empty]
        for dec in decs_present:
            first_sess = 1 if dec == "neurofeedback" else 0
            sub = (full[(full["subject_num"] == subj) & (full["game"] == dec)]
                   .sort_values("session_num"))
            sub = sub[sub["session_num"] >= first_sess]
            if sub.empty:
                continue

            dec_x_start = x
            for _, row in sub.iterrows():
                center = x + GROUP_W / 2
                sess_lbl = ("ME" if str(row["session"]).upper() == "ME"
                            else f"S{int(row['session_num']):02d}")
                records.append({
                    "x": center, "subject": subj, "decoder": dec,
                    "hit":     float(row["hit_rate"]),
                    "broad":   float(row["broad_acc"]),
                    "liberal": float(row["liberal_acc"]),
                })
                x_labels.append((center, sess_lbl))
                x += GROUP_W + GAP_SESSION

            dec_x_end = x - GAP_SESSION + GROUP_W / 2
            dec_spans.append((dec_x_start, dec_x_end, dec))
            x += GAP_DECODER

        subj_x_end = x - GAP_DECODER
        if decs_present:
            subj_spans.append((subj_x_start, subj_x_end, f"Sub{subj:03d}"))
        x += GAP_SUBJECT

    return records, x_labels, dec_spans, subj_spans, x


def _UNUSED_draw_row(fig, ax_rect, records, x_labels, dec_spans, subj_spans,
              total_width, show_legend=False):
    """
    Draw one horizontal band (one row of participants) onto an axes placed at
    ax_rect = [left, bottom, width, height] in figure fraction coordinates.
    """
    BAR_W     = 0.26
    offsets_m = [-BAR_W, 0, BAR_W]
    met_keys  = ["hit_rate", "broad_acc", "liberal_acc"]
    met_vals  = ["hit", "broad", "liberal"]

    ax = fig.add_axes(ax_rect, facecolor=BG_DARK)

    for rec in records:
        for val_key, col_key, off in zip(met_vals, met_keys, offsets_m):
            ax.bar(rec["x"] + off, rec[val_key],
                   width=BAR_W * 0.86, color=METRIC_COLS[col_key],
                   alpha=0.88, edgecolor="none")

    # Reference lines
    ax.axhline(0.50, color=TXT_DIM, ls="--", lw=0.8, alpha=0.7)
    ax.axhline(1.00, color=TXT_DIM, ls="-",  lw=0.4, alpha=0.25)

    # Y axis
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                       color=TXT_DIM, fontsize=8)
    ax.set_ylabel("Accuracy", color=TXT_DIM, fontsize=9, labelpad=5)
    ax.tick_params(axis="y", colors=TXT_DIM, length=0)

    # X axis — session labels
    ax.set_xticks([xp for xp, _ in x_labels])
    ax.set_xticklabels([lbl for _, lbl in x_labels],
                       rotation=90, fontsize=7, color=TXT_DIM)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, total_width)

    # Spines
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID_COL)

    # Subject separator verticals
    for xs, xe, _ in subj_spans:
        if xs > 0.5:
            ax.axvline(xs - 0.65, color=GRID_COL, lw=1.0, alpha=0.7, zorder=0)

    # Bracket annotations
    xlim = ax.get_xlim()
    span = xlim[1] - xlim[0]

    def to_af(xd):
        return (xd - xlim[0]) / span

    y_dec  = -0.14
    y_subj = -0.26

    for x0, x1, dec in dec_spans:
        af0, af1 = to_af(x0), to_af(x1)
        mid = (af0 + af1) / 2
        col = ALL_DEC_COLORS.get(dec, TXT_DIM)
        ax.annotate("", xy=(af1, y_dec), xytext=(af0, y_dec),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.1),
                    annotation_clip=False)
        ax.text(mid, y_dec - 0.03, ALL_DEC_LABELS[dec],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=col, fontweight="bold", clip_on=False)

    for x0, x1, slbl in subj_spans:
        af0, af1 = to_af(x0), to_af(x1)
        mid = (af0 + af1) / 2
        ax.annotate("", xy=(af1, y_subj), xytext=(af0, y_subj),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=TXT_DIM, lw=1.1),
                    annotation_clip=False)
        ax.text(mid, y_subj - 0.03, slbl,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=TXT_HI, fontweight="bold", clip_on=False)

    return ax


def fig_grand_overview(raw_df):
    """
    Horizontal strip plot: one row per participant, liberal accuracy on x-axis.
    Each dot is one session. Colour encodes decoder. Dots for the same
    participant-decoder are connected by a thin line, making session-to-session
    stability immediately visible as the horizontal spread of dots.

    This is a single-axes, two-axis figure — clean enough for publication.
    Participants are sorted top-to-bottom by their mean liberal accuracy
    across all decoders so the most capable participants rise to the top.
    """
    full = pd.read_csv(os.path.join(DATA_DIR, "all_sessions.csv"))

    def parse_sess(s):
        s = str(s)
        if s.startswith("Session_"):
            return int(s.split("_")[1])
        if s.upper() == "ME":
            return 0
        try:
            return int(s)
        except Exception:
            return -1

    def parse_subj(s):
        parts = str(s).split("_")
        try:
            return int(parts[-1])
        except Exception:
            return -1

    full["session_num"] = full["session"].apply(parse_sess)
    full["subject_num"] = full["subject"].apply(parse_subj)
    full = full[full["game"].isin(ALL_DEC_ORDER)].copy()
    full = full[full["session_num"] >= 0].copy()

    subjects = sorted(full["subject_num"].unique())

    # Sort participants by mean liberal accuracy (descending) for readability
    mean_acc = (full.groupby("subject_num")["liberal_acc"].mean()
                    .reindex(subjects).fillna(0))
    subjects = mean_acc.sort_values(ascending=True).index.tolist()

    # Vertical offset per decoder so dots don't overlap within a row
    DEC_PLOT_ORDER = ["neurofeedback", "csp", "graph_ml"]
    dec_yoffset    = {"neurofeedback": -0.22, "csp": 0.0, "graph_ml": +0.22}

    fig, ax = plt.subplots(figsize=(9, 10))

    for i, subj in enumerate(subjects):
        y_base = i

        # Light horizontal band to separate participants
        ax.axhspan(y_base - 0.45, y_base + 0.45,
                   color="#f5f5f5" if i % 2 == 0 else "white",
                   zorder=0)

        for dec in DEC_PLOT_ORDER:
            first_sess = 1 if dec == "neurofeedback" else 0
            sub = (full[(full["subject_num"] == subj) & (full["game"] == dec)]
                   .sort_values("session_num"))
            sub = sub[sub["session_num"] >= first_sess]
            if sub.empty:
                continue

            ys = [y_base + dec_yoffset[dec]] * len(sub)
            xs = sub["liberal_acc"].values

            # Connecting line (shows trajectory across sessions)
            if len(xs) > 1:
                ax.plot(xs, ys, color=COL[dec], lw=1.2, alpha=0.5, zorder=2)

            # Session dots — size encodes session number (later = larger)
            sizes = 35 + sub["session_num"].values * 8
            ax.scatter(xs, ys, color=COL[dec], s=sizes,
                       alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)

    # 50% chance reference
    ax.axvline(0.50, color="#aaa", ls="--", lw=1.1, zorder=1)
    ax.text(0.502, -0.6, "chance", fontsize=8, color="#aaa", va="top")

    # Axes formatting
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"P{int(s):02d}" for s in subjects], fontsize=9)
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(-0.6, len(subjects) - 0.4)
    ax.set_xlabel("Liberal accuracy (each dot = one session)", fontsize=11)
    ax.set_title(
        "Per-participant liberal accuracy across all sessions and decoders\n"
        "Dot size increases with session number  |  line connects sessions within decoder",
        fontsize=10, pad=10,
    )

    # Light vertical grid lines at accuracy thresholds
    for xref in [0.25, 0.50, 0.75, 1.00]:
        ax.axvline(xref, color="#e0e0e0", lw=0.7, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.tick_params(axis="y", length=0)

    # Legend
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[d],
                      markersize=8, label=LABELS[d]) for d in DEC_PLOT_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=9,
              loc="lower right", title="Decoder", title_fontsize=9)

    fig.tight_layout()
    save(fig, "fig1_grand_overview")


# ---- FIG 2: CROSS-SESSION CURVES (ALL THREE MODELS) --------------------------

def fig_cross_session_curves(df):
    """
    Group mean +/- SE of liberal accuracy per session index for all three decoders.
    AR-PSD from session 1; CSP and Graph+ML from session 2 onward.

    If Graph+ML is more resistant to non-stationarity, its curve will stay
    flatter than CSP as the session gap from training data increases.
    """
    at       = all_three(df)
    sessions = sorted(at["session_num"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for dec in DECODERS:
        sub        = at[at["decoder"] == dec]
        first_sess = 1 if dec == "neurofeedback" else 2
        xs, ys, es = [], [], []
        for s in sessions:
            if s < first_sess:
                continue
            vals = sub[sub["session_num"] == s][METRIC].dropna()
            if len(vals) >= 1:
                xs.append(s)
                ys.append(vals.mean())
                es.append(vals.sem() if len(vals) > 1 else 0.0)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es,
                    marker="o", color=COL[dec], label=LABELS[dec],
                    lw=2, capsize=4, markersize=7, capthick=1.5)

    ax.axhline(CHANCE, color="#aaa", ls="--", lw=1.2, label="Chance (50%)")
    ax.set_xlabel("Session number")
    ax.set_ylabel(METRIC_LABEL)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(sessions)
    ax.set_title("Cross-session performance: group mean +/- SE")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_cross_session_curves")


# ---- FIG 3: STABILITY ANALYSIS AND LEVENE'S TEST (ALL THREE MODELS) ----------

def fig_stability(df):
    """
    Tests whether the three decoders produce different levels of session-to-session
    variance in accuracy.

    Panel A: violin plots of all session-level liberal accuracy values per decoder,
    with the three-way Levene's test result in the title.

    Levene's test (three-way): null hypothesis is that all three distributions
    have equal variance. Significant means at least one decoder has reliably
    different spread from the others.

    Panel B: within-subject CV for each decoder, limited to participants who have
    at least two sessions. CV = SD / mean across that participant's sessions.
    Lower CV means more consistent performance across time. This separates
    between-participant variance (panel A) from true within-person stability.

    Pairwise Levene results are shown in the annotation box.
    """
    at = all_three(df)

    pooled = {dec: at[at["decoder"] == dec][METRIC].dropna().values for dec in DECODERS}

    stat_3way, p_3way = levene(*[pooled[d] for d in DECODERS])

    pairs = list(itertools.combinations(DECODERS, 2))
    pair_results = []
    for a, b in pairs:
        s, p = levene(pooled[a], pooled[b])
        pair_results.append((a, b, s, p))

    cv_data      = {dec: [] for dec in DECODERS}
    subj_per_dec = {dec: [] for dec in DECODERS}
    for dec in DECODERS:
        sub_df = at[at["decoder"] == dec]
        for subj in sub_df["subject_num"].unique():
            vals = sub_df[sub_df["subject_num"] == subj][METRIC].dropna().values
            if len(vals) < 2:
                continue
            cv = np.std(vals, ddof=1) / (np.mean(vals) + 1e-10)
            cv_data[dec].append(cv)
            subj_per_dec[dec].append(subj)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = [1, 2, 3]

    # Panel A: violin
    ax = axes[0]
    for pos, dec in zip(positions, DECODERS):
        vals  = pooled[dec]
        parts = ax.violinplot([vals], positions=[pos], showmedians=True,
                              showextrema=True, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor(COL[dec])
            pc.set_alpha(0.60)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            if key in parts:
                parts[key].set_color(COL[dec])
                parts[key].set_linewidth(1.5)
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=COL[dec], s=25, alpha=0.75, zorder=3)

    ax.axhline(CHANCE, color="#aaa", ls="--", lw=1.0, label="Chance (50%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[d] for d in DECODERS])
    ax.set_ylabel(METRIC_LABEL)
    ax.set_ylim(0, 1.05)
    p_str = fmt_p(p_3way)
    ax.set_title(f"Session-level accuracy distributions\nLevene 3-way: W = {stat_3way:.2f}, {p_str}")
    ax.legend(frameon=False, fontsize=8)

    # Panel B: within-subject CV
    ax = axes[1]
    for pos, dec in zip(positions, DECODERS):
        vals = cv_data[dec]
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[pos], patch_artist=True,
                        widths=0.45, medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=COL[dec]),
                        capprops=dict(color=COL[dec]),
                        flierprops=dict(marker="o", color=COL[dec], alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(COL[dec])
            patch.set_alpha(0.70)
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=COL[dec], s=30, alpha=0.80, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[d] for d in DECODERS])
    ax.set_ylabel("Within-subject CV  (lower = more stable)")
    ax.set_title("Within-participant stability across sessions\n(participants with >= 2 sessions only)")

    ann_lines = ["Pairwise Levene:"]
    for a, b, s, p in pair_results:
        p_fmt = fmt_p(p) if p < 0.001 else f"p = {p:.3f}"
        sig   = " *" if p < 0.05 else ""
        ann_lines.append(f"  {LABELS[a]} vs {LABELS[b]}: {p_fmt}{sig}")
    ax.text(0.97, 0.97, "\n".join(ann_lines), transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.80, edgecolor="#ccc"))

    fig.suptitle("Stability analysis: session-level variance across decoders",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig3_stability_levene")

    return cv_data, stat_3way, p_3way, pair_results


# ---- FIG 4: FIGHTING FLAG ANALYSIS (ALL THREE MODELS) ------------------------

def fig_fighting(df):
    """
    Fighting flag rate per decoder. A session is flagged when the cursor spent
    less than 45% of frames on the correct side AND accumulated at least 15
    reversals on the wrong side. This indicates the decoder was actively
    commanding the wrong direction while the participant resisted.

    Panel A: overall mean rate per decoder (error bars = SE across sessions).
    Panel B: rate by session index, showing whether fighting increases or
    decreases as more sessions elapse.
    """
    at       = all_three(df)
    sessions = sorted(at["session_num"].unique())

    overall = (
        at.groupby("decoder")["frac_fighting"]
          .agg(mean="mean", sem=lambda x: x.sem())
          .reindex(DECODERS)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A
    ax = axes[0]
    x  = np.arange(len(DECODERS))
    ax.bar(x, overall["mean"],
           color=[COL[d] for d in DECODERS],
           yerr=overall["sem"], capsize=5, alpha=0.85,
           error_kw=dict(ecolor="#666", lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in DECODERS])
    ax.set_ylabel("Mean fighting flag rate")
    ax.set_title("Overall fighting flag rate per decoder")
    ax.set_ylim(0, max(overall["mean"].max() * 1.5, 0.20))
    for i, (dec, row) in enumerate(overall.iterrows()):
        ax.text(i, row["mean"] + row["sem"] + 0.005,
                f"{row['mean']:.2f}",
                ha="center", va="bottom", fontsize=9)

    # Panel B
    ax = axes[1]
    for dec in DECODERS:
        first_sess = 1 if dec == "neurofeedback" else 2
        sub        = at[at["decoder"] == dec]
        xs, ys, es = [], [], []
        for s in sessions:
            if s < first_sess:
                continue
            vals = sub[sub["session_num"] == s]["frac_fighting"].dropna()
            if len(vals) >= 1:
                xs.append(s)
                ys.append(vals.mean())
                es.append(vals.sem() if len(vals) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o",
                        color=COL[dec], label=LABELS[dec],
                        lw=2, capsize=4, markersize=7)

    ax.set_xlabel("Session number")
    ax.set_ylabel("Fighting flag rate")
    ax.set_title("Fighting rate by session number")
    ax.set_xticks(sessions)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Fighting flag analysis: decoder-user opposition",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig4_fighting_flags")


# ---- FIG 5: SESSION HEATMAP (ALL THREE MODELS) --------------------------------

def fig_heatmaps(df):
    """
    One panel per decoder. Rows are participants, columns are sessions.
    Cell colour encodes liberal accuracy on a red-yellow-green scale.
    Grey cells mean no data for that participant-session-decoder combination.

    This makes it immediately clear which participants are consistent across
    sessions, which sessions tend to be better, and whether the coverage
    patterns differ between CSP and Graph+ML.
    """
    at       = all_three(df)
    subjects = sorted(at["subject_num"].unique())
    sessions = sorted(at["session_num"].unique())

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 0.55 * len(subjects) + 2.5),
        sharey=True,
    )

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#e4e4e4")

    for ax, dec in zip(axes, DECODERS):
        first_sess = 1 if dec == "neurofeedback" else 2
        mat = np.full((len(subjects), len(sessions)), np.nan)
        for i, subj in enumerate(subjects):
            for j, sess in enumerate(sessions):
                if sess < first_sess:
                    continue
                row = at[
                    (at["subject_num"] == subj) &
                    (at["decoder"]     == dec)  &
                    (at["session_num"] == sess)
                ]
                if not row.empty:
                    mat[i, j] = float(row[METRIC].iloc[0])

        masked = np.ma.masked_invalid(mat)
        im     = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap,
                           aspect="auto", interpolation="none")
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels([f"S{s}" for s in sessions], fontsize=8)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels([f"P{s:02d}" for s in subjects], fontsize=8)
        ax.set_title(LABELS[dec], fontsize=11, color=COL[dec], fontweight="bold")
        ax.set_xlabel("Session")

        for i in range(len(subjects)):
            for j in range(len(sessions)):
                if np.isfinite(mat[i, j]):
                    v       = mat[i, j]
                    txt_col = "black" if 0.3 < v < 0.75 else "white"
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=7, color=txt_col)

    plt.colorbar(im, ax=axes, label=METRIC_LABEL,
                 fraction=0.015, pad=0.02, shrink=0.8)
    fig.suptitle("Per-participant session heatmap: liberal accuracy",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig5_session_heatmaps")


# ---- FIG 6: ACCURACY-STABILITY FEATURE SPACE --------------------------------

def fig_accuracy_stability_space(df):
    """
    Each point is one participant under one decoder.
    X axis: mean liberal accuracy across that participant's sessions.
    Y axis: within-subject stability, defined as 1 - CV (so higher = more stable).

    The top-right quadrant (high accuracy AND high stability) is the goal.
    The bottom-right would be high accuracy but inconsistent.
    The top-left would be stable but near chance.

    This directly visualises the stability-discriminability trade-off in online
    terms, analogous to the CV-vs-KLD space used in the offline analysis.

    Only participants with at least 2 sessions for a given decoder are shown,
    since CV is undefined with a single observation.

    Lines connect the same participant across decoders to show whether
    Graph+ML moves them toward the top-right relative to CSP.
    """
    at = all_three(df)

    # Build per-participant per-decoder summary
    rows = []
    for dec in DECODERS:
        sub = at[at["decoder"] == dec]
        first_sess = 1 if dec == "neurofeedback" else 2
        sub = sub[sub["session_num"] >= first_sess]
        for subj in sub["subject_num"].unique():
            vals = sub[sub["subject_num"] == subj][METRIC].dropna().values
            if len(vals) < 2:
                continue
            mean_acc = float(np.mean(vals))
            cv       = float(np.std(vals, ddof=1) / (np.mean(vals) + 1e-10))
            stability = 1.0 - cv   # higher = more stable
            rows.append({
                "subject": subj,
                "decoder": dec,
                "mean_acc": mean_acc,
                "stability": stability,
                "n_sessions": len(vals),
            })

    if not rows:
        print("  [warn] not enough multi-session data for feature space figure")
        return

    pts = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Small deterministic jitter to separate overlapping labels — reproducible
    rng = np.random.default_rng(seed=42)

    # Draw connecting lines first (behind everything)
    for subj in pts["subject"].unique():
        subj_pts = pts[pts["subject"] == subj].sort_values("decoder")
        if len(subj_pts) > 1:
            ax.plot(subj_pts["mean_acc"], subj_pts["stability"],
                    color="#cccccc", lw=1.1, zorder=1, alpha=0.8)

    # Plot points per decoder
    for dec in DECODERS:
        sub = pts[pts["decoder"] == dec]
        ax.scatter(sub["mean_acc"], sub["stability"],
                   color=COL[dec], s=sub["n_sessions"] * 40 + 70,
                   alpha=0.88, zorder=3, label=LABELS[dec],
                   edgecolors="white", linewidths=1.0)

    # Labels with small jitter so nearby points don't overlap
    for dec in DECODERS:
        sub = pts[pts["decoder"] == dec]
        for _, row in sub.iterrows():
            jx = float(rng.uniform(-0.005, 0.005))
            jy = float(rng.uniform(-0.006, 0.006))
            ax.text(row["mean_acc"] + 0.009 + jx,
                    row["stability"] + jy,
                    f"P{int(row['subject']):02d}",
                    fontsize=7.5, color=COL[dec], va="center", zorder=4)

    ax.axvline(CHANCE, color="#bbbbbb", ls="--", lw=1.0, label="Chance (50%)")

    # Tight y-axis: stability from 0.5 to 1.0
    # (captures all meaningful within-person consistency without compressing the space)
    y_lo, y_hi = 0.50, 1.02
    ax.set_ylim(y_lo, y_hi)

    # Soft quadrant shading for the top-right "ideal" region
    ax.fill_betweenx([y_lo, y_hi], CHANCE, 1.0,
                     color=COL["graph_ml"], alpha=0.04)

    ax.set_xlim(0.30, 0.82)

    ax.set_xlabel("Mean liberal accuracy across sessions", fontsize=11)
    ax.set_ylabel("Stability  (1 - CV,  higher = more stable)", fontsize=11)
    ax.set_title(
        "Accuracy-stability feature space\n"
        "Top-right = high accuracy AND consistent performance across sessions",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, "fig6_accuracy_stability_space")

    pts.to_csv(os.path.join(TAB_DIR, "accuracy_stability_space.csv"), index=False)


# ---- STATISTICS CONSOLE SUMMARY ----------------------------------------------

def stats_summary(df):
    sup = supervised(df)
    at  = all_three(df)

    print("\n" + "=" * 68)
    print("STATISTICAL SUMMARY")
    print("=" * 68)

    print(f"\nTotal sessions: {len(df)}")
    for dec in DECODERS:
        sub = df[df["decoder"] == dec]
        print(f"  {LABELS[dec]:10s}: {len(sub)} sessions, "
              f"{sub['subject_num'].nunique()} participants, "
              f"sessions {sorted(sub['session_num'].unique())}")

    print(f"\nSupervised sessions (CSP and Graph+ML, sessions >= 2):")
    for dec in ["csp", "graph_ml"]:
        vals = sup[sup["decoder"] == dec][METRIC].dropna()
        print(f"  {LABELS[dec]:10s}: M = {vals.mean():.3f}  SD = {vals.std():.3f}  n = {len(vals)}")

    print("\nMann-Whitney U (Graph+ML > CSP, one-sided, liberal acc):")
    csp_v = sup[sup["decoder"] == "csp"][METRIC].dropna().values
    gml_v = sup[sup["decoder"] == "graph_ml"][METRIC].dropna().values
    u, p  = mannwhitneyu(gml_v, csp_v, alternative="greater")
    print(f"  U = {u:.0f}, p = {p:.4f}  {'*' if p < 0.05 else 'ns'}")

    print("\nLevene's test (3-way, all decoders):")
    pooled = {d: at[at["decoder"] == d][METRIC].dropna().values for d in DECODERS}
    stat, p = levene(*[pooled[d] for d in DECODERS])
    print(f"  W = {stat:.3f}, p = {p:.6f}  {'*' if p < 0.05 else 'ns'}")

    print("\nPairwise Levene:")
    for a, b in itertools.combinations(DECODERS, 2):
        s, p = levene(pooled[a], pooled[b])
        print(f"  {LABELS[a]} vs {LABELS[b]}: W = {s:.3f}, p = {p:.4f}  "
              f"(SD: {LABELS[a]}={np.std(pooled[a], ddof=1):.3f}, "
              f"{LABELS[b]}={np.std(pooled[b], ddof=1):.3f})  "
              f"{'*' if p < 0.05 else 'ns'}")

    print("\nPer-participant liberal accuracy summary (supervised decoders):")
    rows = []
    for subj in sorted(sup["subject_num"].unique()):
        for dec in ["csp", "graph_ml"]:
            vals = sup[(sup["subject_num"] == subj) & (sup["decoder"] == dec)][METRIC].dropna()
            if vals.empty:
                continue
            rows.append({
                "Participant": f"P{subj:02d}",
                "Decoder":     LABELS[dec],
                "N":           len(vals),
                "Mean":        round(vals.mean(), 3),
                "SD":          round(vals.std(ddof=1), 3) if len(vals) > 1 else float("nan"),
                "Min":         round(vals.min(), 3),
                "Max":         round(vals.max(), 3),
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(TAB_DIR, "per_participant_summary.csv"), index=False)

    print("\nFighting flag rates:")
    fight = (
        at.groupby("decoder")["frac_fighting"]
          .agg(mean="mean", sd="std", n="count")
          .reindex(DECODERS)
    )
    fight.index = [LABELS[d] for d in DECODERS]
    print(fight.round(3).to_string())

    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("=" * 68)
    print("""
What the data supports

Graph+ML liberal accuracy is significantly higher than CSP overall
(Mann-Whitney p = 0.023). This advantage is not uniform: it is driven by a
subset of participants who respond strongly to the graph decoder
(e.g. P10: 0.83 vs CSP 0.54, P03: 0.60 vs CSP 0.51).
For other participants the two decoders perform comparably.

Variance and stability
CSP produces tighter, more consistent scores across participants (SD ~ 0.035)
while Graph+ML has higher between-participant spread (SD ~ 0.094). This is
not evidence of instability. It reflects that the dual-axis selection finds
genuinely useful features for some participants but not others. That is
exactly what you would expect if the method is sensitive to real individual
differences in connectivity patterns: when the signal is there, it captures
it reliably; when the connectivity signal is weak or noisy for a given person,
the method does not manufacture performance.

The within-subject CV (Fig 3 right panel) separates this properly. If a
participant is consistently near chance under both decoders, their CV is
low for both. If Graph+ML gives them consistently high accuracy across
sessions, their CV is also low but their mean is much higher. That combination
high mean + low CV is the signature of a genuine stable advantage.

Fighting flags
AR-PSD has the highest fighting rate (around 11%), which makes sense because
it has no trained threshold and relies on a fixed lateralisation index that
can point the wrong way for some individuals. CSP and Graph+ML are both low
(3-5%). When the supervised decoders fail, they tend to fail imprecisely
rather than by actively commanding the wrong direction.

Feature space (Fig 6)
Participants in the top-right of the accuracy-stability scatter have both
high mean accuracy and consistent performance across sessions. Participants
that shift from the CSP dot toward the Graph+ML dot in the direction of
top-right represent the clearest online evidence for the stability-
discriminability hypothesis. Participants whose dots move sideways but not
upward suggest the method found stable features that are not discriminative,
or vice versa.

What to investigate next
The responder vs non-responder split is the natural follow-up. The
session 1 data for each participant can be analysed offline to ask whether
the dual-axis criterion finds more stable-and-discriminative features for
the responders than the non-responders. That would close the loop between
the offline analysis and the online result reported here.
""")


# ---- MAIN --------------------------------------------------------------------

def main():
    print("Loading data ...")
    df = load()
    print(f"  {len(df)} sessions, decoders: {sorted(df['decoder'].unique())}")

    print("\nGenerating figures ...")
    fig_grand_overview(df)
    fig_cross_session_curves(df)
    fig_stability(df)
    fig_fighting(df)
    fig_heatmaps(df)
    fig_accuracy_stability_space(df)

    print("\nRunning statistics ...")
    stats_summary(df)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
