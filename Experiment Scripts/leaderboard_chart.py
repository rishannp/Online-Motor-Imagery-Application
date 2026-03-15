# generate_accuracy_chart.py
#
# Reads all_sessions.csv (output of analyse_sessions.py) and produces
# bci_accuracy_chart.html in the same directory.
#
# Run: python generate_accuracy_chart.py
# Or:  python generate_accuracy_chart.py path/to/all_sessions.csv

import sys
import os
import json
import csv

# ── game display order within each subject (left to right) ──────────────────
GAME_ORDER = ["neurofeedback", "graph_neurofeedback", "csp", "graph_ml"]
GAME_SHORT = {
    "neurofeedback":       "NFB",
    "graph_neurofeedback": "GNF",
    "csp":                 "CSP",
    "graph_ml":            "GML",
}
GAME_COLOR = {
    "neurofeedback":       "#06b6d4",
    "graph_neurofeedback": "#ec4899",
    "csp":                 "#f59e0b",
    "graph_ml":            "#f97316",
}


def natural_key(s):
    """Sort Subject_0010 after Subject_009 correctly."""
    import re
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "game":    row["game"],
                "subject": row["subject"],
                "session": row["session"],
                "hit":     float(row["hit_rate"]),
                "broad":   float(row["broad_acc"]),
                "liberal": float(row["liberal_acc"]),
            })
    return rows


def build_chart_data(rows):
    """
    Return ordered list of row dicts with subject natural-sorted,
    games ordered NFB → GNF → CSP → GML within each subject,
    sessions sorted within each game.
    """
    # collect unique subjects in natural order
    subjects = sorted(set(r["subject"] for r in rows), key=natural_key)

    ordered = []
    for subj in subjects:
        subj_rows = [r for r in rows if r["subject"] == subj]
        games_present = sorted(
            set(r["game"] for r in subj_rows),
            key=lambda g: GAME_ORDER.index(g) if g in GAME_ORDER else 99
        )
        for game in games_present:
            game_rows = sorted(
                [r for r in subj_rows if r["game"] == game],
                key=lambda r: r["session"]
            )
            for r in game_rows:
                ordered.append(r)

    return ordered, subjects


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BCI Session Accuracy</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:      #0d0f14;
    --surface: #141720;
    --border:  #252a38;
    --label:   #8892a4;
    --label-hi:#c8d0dc;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    font-family: 'IBM Plex Sans', monospace;
    color: var(--label-hi);
    padding: 32px 24px 48px;
    min-height: 100vh;
  }

  header {
    margin-bottom: 32px;
    border-left: 3px solid #60a5fa;
    padding-left: 16px;
  }
  header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #fff;
  }
  header p {
    font-size: 12px;
    color: var(--label);
    margin-top: 4px;
    font-weight: 300;
  }

  .legend {
    display: flex;
    gap: 24px;
    margin-bottom: 28px;
    padding-left: 4px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--label);
  }
  .legend-swatch { width:10px; height:10px; border-radius:2px; flex-shrink:0; }

  .chart-wrap { overflow-x: auto; padding-bottom: 120px; }

  svg text { font-family: 'IBM Plex Mono', monospace; }
</style>
</head>
<body>

<header>
  <h1>BCI SESSION ACCURACY</h1>
  <p>Hit Rate &middot; Broad Accuracy (hit + timeout_close_strong) &middot;
     Liberal Accuracy (hit + strong + weak) &mdash; grouped by subject &amp; game</p>
</header>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#4ade80"></div>Hit Rate</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#60a5fa"></div>Broad Accuracy</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#a78bfa"></div>Liberal Accuracy</div>
</div>

<div class="chart-wrap"><svg id="chart"></svg></div>

<script>
const RAW        = __DATA__;
const GAME_ORDER = __GAME_ORDER__;
const GAME_SHORT = __GAME_SHORT__;
const GAME_COLOR = __GAME_COLOR__;

// ── natural sort on subject string ──────────────────────────────────────────
function naturalKey(s) {
  return s.replace(/(\d+)/g, n => n.padStart(10,'0'));
}

// ── build ordered rows ──────────────────────────────────────────────────────
const subjects = [...new Set(RAW.map(d=>d.subject))].sort((a,b)=>naturalKey(a).localeCompare(naturalKey(b)));

const rows = [];
for (const subj of subjects) {
  const subjRows = RAW.filter(d=>d.subject===subj);
  const gamesPresent = [...new Set(subjRows.map(d=>d.game))]
    .sort((a,b) => {
      const ai = GAME_ORDER.indexOf(a), bi = GAME_ORDER.indexOf(b);
      return (ai===-1?99:ai) - (bi===-1?99:bi);
    });
  for (const game of gamesPresent) {
    const gameRows = subjRows.filter(d=>d.game===game).sort((a,b)=>a.session.localeCompare(b.session));
    for (const r of gameRows) rows.push({...r});
  }
}

// ── layout constants ─────────────────────────────────────────────────────────
const BAR_W       = 6;
const BAR_GAP     = 2;
const GROUP_W     = BAR_W*3 + BAR_GAP*2;   // 22px per session
const SESSION_GAP = 8;
const GAME_GAP    = 16;
const SUBJ_GAP    = 28;
const MARGIN_L    = 52;
const MARGIN_R    = 20;
const MARGIN_T    = 20;
const CHART_H     = 340;
const LABEL_H     = 95;

// ── compute x positions and span metadata ────────────────────────────────────
let x = MARGIN_L;
const posMap   = [];
const subjSpans = [];
const gameSpans = [];

let prevSubj = null, prevGame = null;
let subjStartX = MARGIN_L, gameStartX = MARGIN_L;

for (let i = 0; i < rows.length; i++) {
  const r = rows[i];
  if (i > 0) {
    if (r.subject !== prevSubj) {
      gameSpans.push({subj:prevSubj, game:prevGame, x0:gameStartX, x1:x-1});
      subjSpans.push({subj:prevSubj, x0:subjStartX, x1:x-1});
      x += SUBJ_GAP;
      subjStartX = gameStartX = x;
    } else if (r.game !== prevGame) {
      gameSpans.push({subj:prevSubj, game:prevGame, x0:gameStartX, x1:x-1});
      x += GAME_GAP;
      gameStartX = x;
    } else {
      x += SESSION_GAP;
    }
  }
  posMap.push({i, xc: x + GROUP_W/2});
  x += GROUP_W;
  prevSubj = r.subject;
  prevGame = r.game;
}
gameSpans.push({subj:prevSubj, game:prevGame, x0:gameStartX, x1:x-1});
subjSpans.push({subj:prevSubj, x0:subjStartX, x1:x-1});

const SVG_W = x + MARGIN_R;
const SVG_H = MARGIN_T + CHART_H + LABEL_H;
const yBottom = MARGIN_T + CHART_H;
const yScale  = v => yBottom - v * CHART_H;

// ── SVG helpers ──────────────────────────────────────────────────────────────
const svg = document.getElementById('chart');
svg.setAttribute('width',   SVG_W);
svg.setAttribute('height',  SVG_H);
svg.setAttribute('viewBox', `0 0 ${SVG_W} ${SVG_H}`);

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs={}, parent=svg) {
  const e = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent && parent.appendChild(e);
  return e;
}
function txt(content, attrs={}, parent=svg) {
  const e = el('text', attrs, parent);
  e.textContent = content;
  return e;
}

// ── draw ─────────────────────────────────────────────────────────────────────

// background
el('rect', {x:0,y:0,width:SVG_W,height:SVG_H,fill:'#0d0f14'});

// subject background bands
for (const sp of subjSpans) {
  el('rect', {
    x:sp.x0-6, y:MARGIN_T,
    width:sp.x1-sp.x0+12, height:CHART_H,
    fill:'#141720', rx:3
  });
}

// y-axis gridlines
for (const v of [0, 0.25, 0.5, 0.75, 1.0]) {
  const y = yScale(v);
  el('line', {
    x1:MARGIN_L-8, y1:y, x2:SVG_W-MARGIN_R, y2:y,
    stroke: v===0.5 ? '#3a4060' : '#1e2435',
    'stroke-width': v===0.5 ? 1.5 : 1,
    'stroke-dasharray': v===0.5 ? '4,4' : 'none'
  });
  txt(`${Math.round(v*100)}%`, {
    x:MARGIN_L-10, y:y+4,
    'text-anchor':'end', 'font-size':'9',
    fill: v===0.5 ? '#5a6490' : '#3a4060'
  });
}
txt('chance', {
  x:MARGIN_L-10, y:yScale(0.5)-6,
  'text-anchor':'end', 'font-size':'8', fill:'#3a4060'
});

// game colour bands at top
for (const gs of gameSpans) {
  el('rect', {
    x:gs.x0-4, y:MARGIN_T,
    width:gs.x1-gs.x0+8, height:3,
    fill:GAME_COLOR[gs.game], opacity:'0.85', rx:1
  });
}

// bars
const COLORS  = ['#4ade80','#60a5fa','#a78bfa'];
const METRICS = ['hit','broad','liberal'];

for (const pm of posMap) {
  const r  = rows[pm.i];
  const gx = pm.xc - GROUP_W/2;

  METRICS.forEach((m, mi) => {
    const v  = r[m];
    const bx = gx + mi*(BAR_W+BAR_GAP);
    const by = yScale(v);
    const bh = yBottom - by;

    el('rect', {x:bx, y:by, width:BAR_W, height:bh, fill:COLORS[mi], rx:1, opacity:'0.85'});

    // tooltip
    const g  = el('g');
    el('rect', {x:bx, y:by, width:BAR_W, height:bh, fill:'transparent'}, g);
    const ti = el('title', {}, g);
    ti.textContent = `${r.subject} / ${r.session}\n${r.game}\n${m}: ${Math.round(v*100)}%`;
  });
}

// session labels
const SESSION_LABEL_Y = yBottom + 16;
const BRACKET_GAME_Y  = yBottom + 40;
const BRACKET_SUBJ_Y  = yBottom + 68;

for (const pm of posMap) {
  const r = rows[pm.i];
  txt(r.session.replace('Session_','S'), {
    x:pm.xc, y:SESSION_LABEL_Y,
    'text-anchor':'middle', 'font-size':'8', fill:'#4a5568'
  });
}

// game brackets
for (const gs of gameSpans) {
  const mx = (gs.x0+gs.x1)/2;
  const x0 = gs.x0-4, x1 = gs.x1+4, by = BRACKET_GAME_Y;
  el('path', {
    d:`M${x0},${by-7} L${x0},${by} L${x1},${by} L${x1},${by-7}`,
    fill:'none', stroke:GAME_COLOR[gs.game], 'stroke-width':'1', opacity:'0.7'
  });
  txt(GAME_SHORT[gs.game], {
    x:mx, y:by+11,
    'text-anchor':'middle', 'font-size':'8.5',
    fill:GAME_COLOR[gs.game], 'font-weight':'600'
  });
}

// subject brackets
for (const sp of subjSpans) {
  const mx = (sp.x0+sp.x1)/2;
  const x0 = sp.x0-8, x1 = sp.x1+8, by = BRACKET_SUBJ_Y;
  el('path', {
    d:`M${x0},${by-7} L${x0},${by} L${x1},${by} L${x1},${by-7}`,
    fill:'none', stroke:'#4a5568', 'stroke-width':'1'
  });
  txt(sp.subj.replace('Subject_','Sub'), {
    x:mx, y:by+12,
    'text-anchor':'middle', 'font-size':'8.5',
    fill:'#6b7280', 'font-weight':'600'
  });
}

// y-axis label
const yLabelEl = el('text', {
  x:12, y:MARGIN_T+CHART_H/2,
  'text-anchor':'middle', 'font-size':'9', fill:'#3a4060',
  transform:`rotate(-90,12,${MARGIN_T+CHART_H/2})`
});
yLabelEl.textContent = 'ACCURACY';
</script>
</body>
</html>
"""


def generate_html(rows, subjects):
    data_json       = json.dumps(rows, indent=2)
    game_order_json = json.dumps(GAME_ORDER)
    game_short_json = json.dumps(GAME_SHORT)
    game_color_json = json.dumps(GAME_COLOR)

    html = HTML_TEMPLATE
    html = html.replace('__DATA__',       data_json)
    html = html.replace('__GAME_ORDER__', game_order_json)
    html = html.replace('__GAME_SHORT__', game_short_json)
    html = html.replace('__GAME_COLOR__', game_color_json)
    return html


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "analysis_outputs/all_sessions.csv"

    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        print("Run analyse_sessions.py first, then re-run this script.")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    rows = load_csv(csv_path)
    print(f"  {len(rows)} sessions loaded")

    ordered_rows, subjects = build_chart_data(rows)

    out_dir  = os.path.dirname(os.path.abspath(csv_path))
    out_path = os.path.join(out_dir, "bci_accuracy_chart.html")

    html = generate_html(ordered_rows, subjects)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Chart written: {out_path}")
    print(f"  {len(subjects)} subjects, {len(ordered_rows)} sessions")


if __name__ == "__main__":
    main()
