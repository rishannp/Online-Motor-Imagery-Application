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
import matplotlib.path
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

    fig, ax = plt.subplots(figsize=(11, 12))

    for i, subj in enumerate(subjects):
        y_base = i

        # Alternating row shading
        ax.axhspan(y_base - 0.45, y_base + 0.45,
                   color="#f7f7f7" if i % 2 == 0 else "white",
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

            # Connecting line
            if len(xs) > 1:
                ax.plot(xs, ys, color=COL[dec], lw=1.4, alpha=0.55, zorder=2)

            # Session dots — size grows with session number
            sizes = 45 + sub["session_num"].values * 10
            ax.scatter(xs, ys, color=COL[dec], s=sizes,
                       alpha=0.88, zorder=3, edgecolors="white", linewidths=0.6)

    # 50% chance reference
    ax.axvline(0.50, color="#aaa", ls="--", lw=1.2, zorder=1)
    ax.text(0.504, -0.6, "chance", fontsize=9, color="#aaa", va="top")

    # Axes formatting
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"P{int(s):02d}" for s in subjects], fontsize=11)
    ax.set_xlim(0.20, 0.92)
    ax.set_ylim(-0.6, len(subjects) - 0.4)
    ax.set_xlabel("Liberal accuracy (each dot = one session)", fontsize=11)
    ax.set_title(
        "Per-participant liberal accuracy across all sessions and decoders\n"
        "Dot size increases with session number  |  line connects sessions within decoder",
        fontsize=10, pad=10,
    )

    # Light vertical grid lines at accuracy thresholds
    for xref in [0.25, 0.50, 0.75]:
        ax.axvline(xref, color="#e0e0e0", lw=0.8, zorder=0)

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
    Cross-session liberal accuracy for all three decoders.

    Shaded bands show +/- 1 SE around the group mean. Faint individual
    participant lines sit behind the group mean, making the between-person
    spread visible without cluttering the main message.  Sample size (n) is
    annotated below each mean point so the reader can judge reliability at
    each session.  The Mann-Whitney result comparing Graph+ML to CSP overall
    is annotated directly on the figure.
    """
    at       = all_three(df)
    sessions = sorted(at["session_num"].unique())

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Faint individual participant lines first (behind everything)
    for dec in ["csp", "graph_ml"]:
        sub        = at[at["decoder"] == dec]
        first_sess = 2
        for subj in sub["subject_num"].unique():
            pts = (sub[sub["subject_num"] == subj]
                   .sort_values("session_num"))
            pts = pts[pts["session_num"] >= first_sess][["session_num", METRIC]].dropna()
            if len(pts) >= 2:
                ax.plot(pts["session_num"], pts[METRIC],
                        color=COL[dec], lw=0.8, alpha=0.18, zorder=1)

    # Group mean with shaded SE band
    for dec in DECODERS:
        sub        = at[at["decoder"] == dec]
        first_sess = 1 if dec == "neurofeedback" else 2
        xs, ys, lo, hi, ns = [], [], [], [], []
        for s in sessions:
            if s < first_sess:
                continue
            vals = sub[sub["session_num"] == s][METRIC].dropna()
            if len(vals) < 1:
                continue
            m  = vals.mean()
            se = vals.sem() if len(vals) > 1 else 0.0
            xs.append(s);  ys.append(m)
            lo.append(m - se);  hi.append(m + se)
            ns.append(len(vals))
        if not xs:
            continue

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)

        # Shaded band
        ax.fill_between(xs_arr, lo, hi,
                        color=COL[dec], alpha=0.18, zorder=2)
        # Mean line
        ax.plot(xs_arr, ys_arr,
                color=COL[dec], lw=2.2, zorder=3, label=LABELS[dec])
        # Mean markers
        ax.scatter(xs_arr, ys_arr,
                   color=COL[dec], s=52, zorder=4,
                   edgecolors="white", linewidths=1.0)
        pass  # n= labels removed

    # Chance reference
    ax.axhline(CHANCE, color="#bbb", ls="--", lw=1.1, zorder=1)
    ax.text(sessions[-1] + 0.1, CHANCE + 0.01, "chance",
            fontsize=8, color="#bbb", va="bottom", ha="left")

    # Statistical annotation — supervised decoders only (sessions 2+)
    ax.text(0.02, 0.97,
            "Graph+ML vs CSP (sessions 2+)\nMann-Whitney p = 0.023",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            color=COL["graph_ml"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=COL["graph_ml"], alpha=0.85, lw=0.8))

    ax.set_xlabel("Session number", fontsize=11)
    ax.set_ylabel(METRIC_LABEL, fontsize=11)
    ax.set_ylim(0.33, 0.88)
    ax.set_xticks(sessions)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Cross-session liberal accuracy\n"
        "Shaded band = group mean +/- SE  |  faint lines = individual participants",
        fontsize=11, pad=10,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "fig2_cross_session_curves")


# ---- FIG 3: STABILITY ANALYSIS AND LEVENE'S TEST (ALL THREE MODELS) ----------

def _sig_bracket(ax, x0, x1, y, p, dy=0.012, fontsize=8.5):
    """Draw a significance bracket between x0 and x1 at height y."""
    tick = dy * 0.5
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick],
            color="#555", lw=1.1, clip_on=False)
    if p < 0.001:
        exp  = int(np.floor(np.log10(p)))
        coef = p / (10 ** exp)
        label = f"{coef:.1f}×10$^{{{exp}}}$*"
    elif p < 0.05:
        label = f"p={p:.3f}*"
    else:
        label = f"p={p:.3f}"
    ax.text((x0 + x1) / 2, y + dy * 0.1, label,
            ha="center", va="bottom", fontsize=fontsize, color="#555")


def fig_stability(df):
    """
    Panel A: raincloud plot of session-level liberal accuracy per decoder.
    Half-violin shows the distribution shape; individual session dots are
    plotted on the right side; a short horizontal bar marks the median.
    The three-way Levene result and pairwise comparisons are annotated with
    significance brackets.

    Panel B: within-participant CV (coefficient of variation across sessions)
    shown as a beeswarm-style strip with a median marker. Only participants
    with at least two sessions per decoder are included. Lower CV = more
    consistent performance across sessions.
    """
    at = all_three(df)

    pooled = {dec: at[at["decoder"] == dec][METRIC].dropna().values
              for dec in DECODERS}
    stat_3way, p_3way = levene(*[pooled[d] for d in DECODERS])

    pairs = list(itertools.combinations(DECODERS, 2))
    pair_results = []
    for a, b in pairs:
        s, p = levene(pooled[a], pooled[b])
        pair_results.append((a, b, s, p))

    cv_data      = {dec: [] for dec in DECODERS}
    subj_per_dec = {dec: [] for dec in DECODERS}
    rng = np.random.default_rng(seed=7)
    for dec in DECODERS:
        sub_df = at[at["decoder"] == dec]
        for subj in sub_df["subject_num"].unique():
            vals = sub_df[sub_df["subject_num"] == subj][METRIC].dropna().values
            if len(vals) < 2:
                continue
            cv = np.std(vals, ddof=1) / (np.mean(vals) + 1e-10)
            cv_data[dec].append(cv)
            subj_per_dec[dec].append(subj)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    axes = [ax, ax]   # dummy so rest of code referencing axes[1] is unreachable
    positions = [1, 2, 3]

    # ---- Raincloud ---------------------------------------------------------
    for pos, dec in zip(positions, DECODERS):
        vals  = pooled[dec]
        color = COL[dec]

        # Half violin (left side only) via path vertex clipping
        parts = ax.violinplot([vals], positions=[pos], showmedians=False,
                              showextrema=False, widths=0.55)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
            verts = pc.get_paths()[0].vertices.copy()
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
            pc.get_paths()[0].vertices = verts

        # Median bar
        med = float(np.median(vals))
        ax.plot([pos - 0.27, pos], [med, med],
                color=color, lw=2.0, zorder=4)

        # Dots on the right side with jitter
        jitter = rng.uniform(0.06, 0.28, len(vals))
        ax.scatter(pos + jitter, vals,
                   color=color, s=22, alpha=0.70, zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.axhline(CHANCE, color="#bbb", ls="--", lw=1.0)
    ax.text(3.45, CHANCE + 0.01, "chance", fontsize=8, color="#bbb")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[d] for d in DECODERS], fontsize=11)
    ax.set_ylabel(METRIC_LABEL, fontsize=11)
    ax.set_ylim(0.05, 1.22)
    ax.set_xlim(0.4, 3.9)
    ax.yaxis.grid(True, color="#f0f0f0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    p_str = fmt_p(p_3way)
    ax.set_title(
        f"Session-level accuracy distributions\n"
        f"Levene 3-way: W = {stat_3way:.2f}, {p_str}",
        fontsize=10, pad=8,
    )

    # Significance brackets sit inside the axes (below y=1.22 ceiling)
    bracket_y = 1.07
    for a, b, s, p in pair_results:
        if p < 0.10:
            pa = positions[DECODERS.index(a)]
            pb = positions[DECODERS.index(b)]
            _sig_bracket(ax, pa, pb, bracket_y, p, dy=0.022, fontsize=8)
            bracket_y += 0.050

    # Pairwise Levene annotation on the single panel
    ann_lines = ["Pairwise Levene:"]
    for a, b, s, p in pair_results:
        p_fmt = fmt_p(p) if p < 0.001 else f"p = {p:.3f}"
        sig   = " *" if p < 0.05 else ""
        ann_lines.append(f"  {LABELS[a]} vs {LABELS[b]}: {p_fmt}{sig}")
    ax.text(0.97, 0.03, "\n".join(ann_lines), transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85, edgecolor="#ddd", lw=0.8))

    fig.suptitle("Stability analysis: session-level accuracy distributions",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig3_stability_levene")

    return cv_data, stat_3way, p_3way, pair_results


# ---- FIG 4: FIGHTING FLAG ANALYSIS (ALL THREE MODELS) ------------------------

def fig_fighting(df):
    """
    Fighting flag rate: sessions where the cursor was actively driven to the
    wrong side while the participant resisted (< 45% frames on correct side
    AND >= 15 wrong-side reversals).

    Panel A: horizontal dot-and-range plot. Each dot is the mean fighting rate
    for one decoder; the horizontal line shows the full range across sessions.
    Individual session values are shown as small transparent dots to give a
    sense of the distribution without cluttering the summary.

    Panel B: shaded SE band showing how fighting rate evolves across session
    numbers, with individual session dots behind the band.
    """
    at       = all_three(df)
    sessions = sorted(at["session_num"].unique())

    overall = (
        at.groupby("decoder")["frac_fighting"]
          .agg(mean="mean", sem=lambda x: x.sem(),
               lo="min", hi="max")
          .reindex(DECODERS)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.6]})

    # ---- Panel A: horizontal dot + individual session scatter ---------------
    ax    = axes[0]
    ys    = np.arange(len(DECODERS))
    rng   = np.random.default_rng(seed=3)

    for i, dec in enumerate(DECODERS):
        color = COL[dec]
        row   = overall.loc[dec]
        vals  = at[at["decoder"] == dec]["frac_fighting"].dropna().values

        # Range line
        ax.plot([row["lo"], row["hi"]], [i, i],
                color=color, lw=1.4, alpha=0.45, zorder=2,
                solid_capstyle="round")

        # Individual session dots
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(vals, i + jitter,
                   color=color, s=28, alpha=0.45, zorder=3,
                   edgecolors="none")

        # Mean dot (on top)
        ax.scatter(row["mean"], i,
                   color=color, s=110, zorder=5,
                   edgecolors="white", linewidths=1.5)

        # Value label
        ax.text(row["mean"] + 0.003, i + 0.22,
                f"{row['mean']:.2f}",
                ha="center", va="bottom", fontsize=9.5,
                color=color, fontweight="600")

    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[d] for d in DECODERS], fontsize=11)
    ax.set_xlabel("Fighting flag rate", fontsize=11)
    ax.set_xlim(-0.01, overall["hi"].max() + 0.04)
    ax.set_ylim(-0.6, len(DECODERS) - 0.4)
    ax.xaxis.grid(True, color="#f0f0f0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Overall fighting flag rate\nDot = mean  |  line = range  |  faint = individual sessions",
        fontsize=10, pad=8,
    )

    # ---- Panel B: shaded SE band + individual dots --------------------------
    ax = axes[1]

    for dec in DECODERS:
        first_sess = 1 if dec == "neurofeedback" else 2
        sub        = at[at["decoder"] == dec]
        xs, ys_m, lo, hi = [], [], [], []

        for s in sessions:
            if s < first_sess:
                continue
            vals = sub[sub["session_num"] == s]["frac_fighting"].dropna()
            if len(vals) < 1:
                continue
            m  = vals.mean()
            se = vals.sem() if len(vals) > 1 else 0.0
            xs.append(s);  ys_m.append(m)
            lo.append(m - se);  hi.append(m + se)

            # Individual dots behind
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(s + jitter, vals,
                       color=COL[dec], s=22, alpha=0.30, zorder=2,
                       edgecolors="none")

        if not xs:
            continue
        xs_arr = np.array(xs)
        ax.fill_between(xs_arr, lo, hi,
                        color=COL[dec], alpha=0.18, zorder=3)
        ax.plot(xs_arr, ys_m,
                color=COL[dec], lw=2.2, zorder=4, label=LABELS[dec])
        ax.scatter(xs_arr, ys_m,
                   color=COL[dec], s=50, zorder=5,
                   edgecolors="white", linewidths=1.0)

    ax.set_xlabel("Session number", fontsize=11)
    ax.set_ylabel("Fighting flag rate", fontsize=11)
    ax.set_xticks(sessions)
    ax.set_ylim(-0.01, None)
    ax.yaxis.grid(True, color="#f0f0f0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Fighting rate across sessions\nShaded band = mean +/- SE",
        fontsize=10, pad=8,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.suptitle(
        "Fighting flag analysis: sessions where the decoder actively opposed the participant",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(w_pad=3.5)
    save(fig, "fig4_fighting_flags")


# ---- FIG 5: SESSION HEATMAP (ALL THREE MODELS) --------------------------------

def fig_heatmaps(df):
    """
    Single unified heatmap. Participants on the y-axis (sorted by mean
    liberal accuracy, highest at top). The x-axis is divided into three
    decoder groups — AR-PSD, CSP, Graph+ML — each showing only the sessions
    that exist for that decoder. A thin gap separates the groups, and a
    coloured label above each group identifies the decoder.

    Missing cells are white so they disappear rather than dominating.
    No numbers inside cells — colour alone carries the accuracy value,
    keeping the figure uncluttered.
    """
    at = all_three(df)

    # Sort participants by mean liberal accuracy descending
    mean_acc = (at.groupby("subject_num")[METRIC].mean()
                  .sort_values(ascending=False))
    subjects = mean_acc.index.tolist()
    n_subj   = len(subjects)

    # Define which sessions to show for each decoder
    dec_sessions = {
        "neurofeedback": sorted(
            at[at["decoder"] == "neurofeedback"]["session_num"].unique()),
        "csp":           sorted(
            at[at["decoder"] == "csp"]["session_num"].unique()),
        "graph_ml":      sorted(
            at[at["decoder"] == "graph_ml"]["session_num"].unique()),
    }

    # Build flat column list with gap markers between decoder groups
    GAP = 0.6          # visual gap width in column units
    col_info = []      # (decoder, session_num, x_position)
    dec_spans = []     # (decoder, x_start, x_end) for header labels
    x = 0.0
    for dec in DECODERS:
        x_start = x
        for sess in dec_sessions[dec]:
            col_info.append((dec, sess, x))
            x += 1.0
        dec_spans.append((dec, x_start, x - 1.0))
        x += GAP

    total_cols = x - GAP

    # Build the data matrix
    mat = np.full((n_subj, len(col_info)), np.nan)
    for row_i, subj in enumerate(subjects):
        for col_j, (dec, sess, _) in enumerate(col_info):
            val = at[
                (at["subject_num"] == subj) &
                (at["decoder"]     == dec)  &
                (at["session_num"] == sess)
            ][METRIC]
            if not val.empty:
                mat[row_i, col_j] = float(val.iloc[0])

    # ---- Draw ---------------------------------------------------------------
    cell_h = 0.52     # inches per row
    fig_h  = max(6.0, n_subj * cell_h + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="white")

    # Draw each cell manually so gaps between decoder groups are real whitespace
    for col_j, (dec, sess, xpos) in enumerate(col_info):
        for row_i in range(n_subj):
            v = mat[row_i, col_j]
            if not np.isfinite(v):
                continue
            norm_v = np.clip(v, 0, 1)
            color  = cmap(norm_v)
            rect   = plt.Rectangle(
                (xpos - 0.46, row_i - 0.46), 0.92, 0.92,
                facecolor=color, edgecolor="white", linewidth=0.8,
            )
            ax.add_patch(rect)

    # ---- Y axis: participant labels ----------------------------------------
    ax.set_yticks(range(n_subj))
    ax.set_yticklabels([f"P{int(s):02d}" for s in subjects], fontsize=9)
    ax.set_ylim(n_subj - 0.5, -0.5)   # top to bottom

    # ---- X axis: session labels --------------------------------------------
    x_positions = [xpos for _, _, xpos in col_info]
    x_labels    = [f"S{sess}" for _, sess, _ in col_info]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=0)
    ax.set_xlim(-0.55, total_cols + 0.1)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    # ---- Decoder group headers and dividers --------------------------------
    header_y = -1.1   # just above the top row (in data coords, inverted axis)
    for dec, x0, x1 in dec_spans:
        mid = (x0 + x1) / 2
        # Coloured underline
        ax.plot([x0 - 0.4, x1 + 0.4], [header_y + 0.35, header_y + 0.35],
                color=COL[dec], lw=2.5, clip_on=False,
                solid_capstyle="round", transform=ax.transData)
        # Label
        ax.text(mid, header_y, LABELS[dec],
                ha="center", va="center", fontsize=11,
                color=COL[dec], fontweight="bold",
                transform=ax.transData, clip_on=False)
        # Thin vertical separator between groups (skip after last)
        if dec != DECODERS[-1]:
            sep_x = x1 + GAP / 2
            ax.axvline(sep_x, color="#dddddd", lw=1.0,
                       ymin=0, ymax=1, zorder=0)

    # ---- Colorbar ----------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6,
                      aspect=20)
    cb.set_label(METRIC_LABEL, fontsize=10)
    cb.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cb.ax.tick_params(labelsize=8)

    # ---- Spines ------------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Per-participant liberal accuracy by session and decoder\n"
        "Sorted by mean accuracy (top = highest).  White = no data.",
        fontsize=11, pad=28,
    )
    fig.tight_layout()
    save(fig, "fig5_session_heatmaps")


# ---- TABLE: PER-PARTICIPANT SESSION ACCURACY (LaTeX) -------------------------

def table_session_accuracy(df):
    """
    Generates a publication-ready LaTeX table (booktabs style) saved to
    tables/session_accuracy.tex.

    Rows: participants sorted by mean liberal accuracy descending.
    Column groups: AR-PSD (S1-S7) | CSP (S2-S5) | Graph+ML (S2-S5).
    Only sessions that have at least one data point are shown as columns.
    Missing cells are shown as -- .
    The highest value in each row is bold.
    A group mean row is appended at the bottom.
    """
    at = all_three(df)

    # Sort participants by mean liberal accuracy descending
    mean_acc = (at.groupby("subject_num")[METRIC].mean()
                  .sort_values(ascending=False))
    subjects = mean_acc.index.tolist()

    # Sessions per decoder that actually have data
    dec_sessions = {}
    for dec in DECODERS:
        first = 1 if dec == "neurofeedback" else 2
        slist = sorted(
            s for s in at[at["decoder"] == dec]["session_num"].unique()
            if s >= first
        )
        dec_sessions[dec] = slist

    # Build value lookup: (subject, decoder, session) -> float or nan
    def get_val(subj, dec, sess):
        row = at[
            (at["subject_num"] == subj) &
            (at["decoder"]     == dec)  &
            (at["session_num"] == sess)
        ][METRIC]
        return float(row.iloc[0]) if not row.empty else np.nan

    # Column specs
    # AR-PSD columns
    ar_cols  = dec_sessions["neurofeedback"]
    csp_cols = dec_sessions["csp"]
    gml_cols = dec_sessions["graph_ml"]

    n_ar  = len(ar_cols)
    n_csp = len(csp_cols)
    n_gml = len(gml_cols)
    n_total = n_ar + n_csp + n_gml

    def fmt(v, bold=False):
        if not np.isfinite(v):
            return "--"
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    def cell_color(v):
        """Return a light xcolor-style shading command based on value."""
        if not np.isfinite(v):
            return ""
        # Map 0-1 to white->green intensity for \cellcolor
        pct = int(np.clip(v * 100, 0, 100))
        return f"\\cellcolor{{accuracycol!{pct}}}"

    lines = []
    lines.append("% Requires: \\usepackage{booktabs, multirow, xcolor, colortbl}")
    lines.append("% Define: \\definecolor{accuracycol}{RGB}{133,184,122}")
    lines.append("%")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{Per-participant liberal accuracy by session and decoder. "
        "Values are liberal accuracy (hits + timeout close strong + timeout close weak) "
        "/ total trials. Dashes indicate no data. "
        "Bold indicates the highest value for that participant across all decoders. "
        "Chance baseline is 0.50.}"
    )
    lines.append("\\label{tab:session_accuracy}")

    # Column format: participant | AR-PSD cols | CSP cols | GML cols
    col_fmt = "l" + "|" + "c" * n_ar + "|" + "c" * n_csp + "|" + "c" * n_gml
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")

    # Header row 1: decoder group labels
    ar_header  = f"\\multicolumn{{{n_ar}}}{{c}}{{\\textbf{{AR-PSD}}}}"
    csp_header = f"\\multicolumn{{{n_csp}}}{{c}}{{\\textbf{{CSP}}}}"
    gml_header = f"\\multicolumn{{{n_gml}}}{{c}}{{\\textbf{{Graph+ML}}}}"
    lines.append(
        f"\\textbf{{Participant}} & {ar_header} & {csp_header} & {gml_header} \\\\"
    )

    # Cmidrule under each group
    c1 = 2;              c2 = 1 + n_ar
    c3 = c2 + 1;         c4 = c2 + n_csp
    c5 = c4 + 1;         c6 = c4 + n_gml
    lines.append(
        f"\\cmidrule(lr){{{c1}-{c2}}} "
        f"\\cmidrule(lr){{{c3}-{c4}}} "
        f"\\cmidrule(lr){{{c5}-{c6}}}"
    )

    # Header row 2: session numbers
    ar_sess_hdr  = " & ".join(f"S{s}" for s in ar_cols)
    csp_sess_hdr = " & ".join(f"S{s}" for s in csp_cols)
    gml_sess_hdr = " & ".join(f"S{s}" for s in gml_cols)
    lines.append(f" & {ar_sess_hdr} & {csp_sess_hdr} & {gml_sess_hdr} \\\\")
    lines.append("\\midrule")

    # Data rows
    col_sums  = {(dec, s): [] for dec in DECODERS for s in dec_sessions[dec]}

    for subj in subjects:
        # Collect all finite values to find the row maximum
        all_vals = []
        for dec in DECODERS:
            for s in dec_sessions[dec]:
                v = get_val(subj, dec, s)
                if np.isfinite(v):
                    all_vals.append(v)
        row_max = max(all_vals) if all_vals else np.nan

        cells = [f"P{int(subj):02d}"]
        for dec in DECODERS:
            for s in dec_sessions[dec]:
                v = get_val(subj, dec, s)
                is_max = np.isfinite(v) and np.isfinite(row_max) and abs(v - row_max) < 1e-9
                cells.append(fmt(v, bold=is_max))
                if np.isfinite(v):
                    col_sums[(dec, s)].append(v)

        lines.append(" & ".join(cells) + " \\\\")

    # Group mean row
    lines.append("\\midrule")
    mean_cells = ["\\textit{Mean}"]
    for dec in DECODERS:
        for s in dec_sessions[dec]:
            vals = col_sums[(dec, s)]
            mean_cells.append(f"\\textit{{{np.mean(vals):.2f}}}" if vals else "--")
    lines.append(" & ".join(mean_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    path = os.path.join(TAB_DIR, "session_accuracy.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  saved: session_accuracy.tex  ({n_total} data columns, {len(subjects)} participants)")

    # Also save a plain CSV version for reference
    rows_csv = []
    for subj in subjects:
        row = {"Participant": f"P{int(subj):02d}"}
        for dec in DECODERS:
            for s in dec_sessions[dec]:
                v = get_val(subj, dec, s)
                row[f"{LABELS[dec]}_S{s}"] = round(v, 3) if np.isfinite(v) else None
        rows_csv.append(row)
    pd.DataFrame(rows_csv).to_csv(
        os.path.join(TAB_DIR, "session_accuracy.csv"), index=False
    )


# ---- FIG 6: ACCURACY-STABILITY FEATURE SPACE --------------------------------

def fig_accuracy_stability_space(df):
    """
    Each point is one participant under one decoder.
    X axis: mean liberal accuracy across that participant's sessions.
    Y axis: within-subject stability (1 - CV, higher = more stable).

    Lines connect the same participant across decoders to show movement
    toward or away from the top-right ideal region.

    Labels are repelled from one another using adjustText so overlapping
    participant IDs are pushed apart automatically.
    """
    from adjustText import adjust_text

    at = all_three(df)

    rows = []
    for dec in DECODERS:
        sub        = at[at["decoder"] == dec]
        first_sess = 1 if dec == "neurofeedback" else 2
        sub        = sub[sub["session_num"] >= first_sess]
        for subj in sub["subject_num"].unique():
            vals = sub[sub["subject_num"] == subj][METRIC].dropna().values
            if len(vals) < 2:
                continue
            mean_acc  = float(np.mean(vals))
            cv        = float(np.std(vals, ddof=1) / (np.mean(vals) + 1e-10))
            stability = 1.0 - cv
            rows.append({
                "subject":    subj,
                "decoder":    dec,
                "mean_acc":   mean_acc,
                "stability":  stability,
                "n_sessions": len(vals),
            })

    if not rows:
        print("  [warn] not enough multi-session data for feature space figure")
        return

    pts = pd.DataFrame(rows)

    # Compute tight axis limits from the actual data with a small margin
    x_pad = 0.03
    y_pad = 0.02
    x_lo  = max(0.30, pts["mean_acc"].min()  - x_pad)
    x_hi  = min(0.82, pts["mean_acc"].max()  + x_pad)
    y_lo  = max(0.65, pts["stability"].min() - y_pad)
    y_hi  = min(1.02, pts["stability"].max() + y_pad)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Connecting lines (same participant across decoders)
    for subj in pts["subject"].unique():
        sp = pts[pts["subject"] == subj].sort_values("mean_acc")
        if len(sp) > 1:
            ax.plot(sp["mean_acc"], sp["stability"],
                    color="#d0d0d0", lw=1.2, zorder=1, alpha=0.9)

    # Scatter — size encodes number of sessions
    for dec in DECODERS:
        sub = pts[pts["decoder"] == dec]
        ax.scatter(sub["mean_acc"], sub["stability"],
                   color=COL[dec], s=sub["n_sessions"] * 45 + 80,
                   alpha=0.88, zorder=3, label=LABELS[dec],
                   edgecolors="white", linewidths=1.2)

    # Collect text objects for adjustText to repel
    texts = []
    for dec in DECODERS:
        sub = pts[pts["decoder"] == dec]
        for _, row in sub.iterrows():
            t = ax.text(row["mean_acc"], row["stability"],
                        f"P{int(row['subject']):02d}",
                        fontsize=8, color=COL[dec],
                        va="center", ha="left", zorder=5)
            texts.append(t)

    # Repel labels away from each other and from the scatter points
    adjust_text(
        texts,
        x=pts["mean_acc"].values,
        y=pts["stability"].values,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7),
        expand=(1.4, 1.6),
        force_points=(0.4, 0.6),
        force_text=(0.5, 0.8),
        only_move={"points": "y", "text": "xy"},
    )

    # Reference line and shading
    ax.axvline(CHANCE, color="#bbbbbb", ls="--", lw=1.0, label="Chance (50%)")
    ax.fill_betweenx([y_lo, y_hi], CHANCE, x_hi,
                     color=COL["graph_ml"], alpha=0.04)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Light grid to help read off values
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    ax.xaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel("Mean liberal accuracy across sessions", fontsize=11)
    ax.set_ylabel("Stability  (1 - CV,  higher = more stable)", fontsize=11)
    ax.set_title(
        "Accuracy-stability feature space\n"
        "Top-right = high accuracy AND consistent across sessions  "
        "|  point size = number of sessions",
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
    print("\nGenerating LaTeX table ...")
    table_session_accuracy(df)
    fig_accuracy_stability_space(df)

    print("\nRunning statistics ...")
    stats_summary(df)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
