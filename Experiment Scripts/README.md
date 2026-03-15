# Online Motor Imagery BCI Decoder

A comparative framework for online EEG-based motor imagery (MI) brain-computer interfaces (BCIs), built to investigate whether **graph-based connectivity features improve decoding robustness under non-stationarity** — particularly when classifiers are trained exclusively on data from prior sessions.

---

## Research Question

> *Are graph-based BCIs better at handling EEG non-stationarity? Do they improve decoding accuracy when trained only on data from previous sessions?*

Four BCI systems are compared, ranging from a simple spectral baseline to a novel graph-based approach that selects features by their discriminability and temporal stability:

| # | System | Folder | Feature Type | Classifier |
|---|--------|--------|-------------|------------|
| 1 | **AR-PSD (Alpha-Asymmetry)** | `neurofeedback/` | Hjorth Laplacian → AR power spectrum | Threshold |
| 2 | **CSP** | `csp/` | Common Spatial Pattern log-variance | SVM (linear) |
| 3 | **Graph Asymmetry** | `graph_neurofeedback/` | PLV / ImCoh motor asymmetry | Threshold |
| 4 | **Graph + ML** *(novel)* | `graph_ml/` | PLV graph features filtered by CV + KL divergence | SVM / LogReg / KNN |

All four systems share the same underlying EEG hardware interface (LSL), the same pygame game interface, and the same trial recording format, enabling direct cross-system comparison.

---

## Project Structure

```
Experiment Scripts/
│
├── neurofeedback/           # BCI 1: AR-PSD Alpha-Asymmetry baseline
├── csp/                     # BCI 2: Common Spatial Pattern + SVM
├── graph_neurofeedback/     # BCI 3: Graph Connectivity Asymmetry
├── graph_ml/                # BCI 4: Graph + ML (novel stability-filtered features)
│
├── global_accuracy_tracks.py    # Post-hoc analysis: loads all session_data.pkl, exports CSV
├── chance_level_simulation.py   # Permutation-based chance-level estimation
├── leaderboard_chart.py         # Cross-subject performance visualisation
└── analysis_outputs/            # Generated CSV reports
```

Each BCI module contains:

```
<module>/
├── config.py            # Subject ID, session ID, training source, hyperparameters
├── main.py              # Entry point: trains model, launches online BCI loop + game
├── *_pipeline.py        # Online inference class (feature extraction → binary command)
├── *_training.py        # Offline training: load trials, fit model, save .pkl pack
├── preprocess.py        # EEG bandpass filter, channel selection
├── game.py              # pygame trial loop, cursor game, trial recording
├── lsl_stream.py        # LSL inlet (live hardware or simulator)
├── sim_lsl_mi.py        # LSL simulator: replays recorded session_data.pkl
├── trained_models/      # Saved classifier packs (per subject/session)
└── training_results/    # Recorded trial data: session_data.pkl
```

---

## The Four BCI Systems

### BCI 1 — AR-PSD Alpha-Asymmetry (`neurofeedback/`)

A neuroscientifically motivated, training-free baseline based on the lateralisation of motor cortex alpha power.

**Pipeline:**
1. Apply Hjorth surface Laplacian on C3 and C4 (subtract mean of 4 neighbours)
2. Fit a Yule–Walker AR model (order 12) on each 2-second window
3. Integrate AR-estimated PSD in the alpha band (10–13 Hz)
4. Control signal: `C4_power − C3_power`
5. Z-score using a rolling 30 s baseline; clip to ±3; threshold at zero

**Key properties:** No training required. Subject-agnostic. Sensitive to volume conduction and to individual alpha frequency variation.

---

### BCI 2 — CSP (`csp/`)

The standard supervised approach for MI decoding, trained offline on data from prior sessions.

**Pipeline:**
1. Bandpass filter EEG (8–30 Hz, causal IIR order 4)
2. Fit CSP spatial filters (6 components) on labelled trial windows from training sessions
3. Extract log-variance of CSP-filtered signals per 1-second window
4. Train linear SVM (grid search over C ∈ {0.1, 1.0, 10.0}, class-balanced)
5. Online: sliding window → SVM decision margin → 15-vote majority smooth → binary command

**Key properties:** Well-validated method. Sensitive to covariate shift between sessions (non-stationarity).

---

### BCI 3 — Graph Asymmetry (`graph_neurofeedback/`)

A training-free graph-based system using inter-electrode phase synchrony to measure motor cortex lateralisation.

**Pipeline:**
1. Compute a connectivity matrix between all motor-region electrodes using either:
   - **PLV** (Phase Locking Value): `|mean(exp(i·Δφ))|` — sensitive to volume conduction
   - **ImCoh** (Imaginary Coherence): `Im(Cxy) / √(Cxx·Cyy)` — immune to zero-lag coupling
2. Compute asymmetry score:
   ```
   asym = (intra_right − intra_left) / (cross + ε)
   ```
   where `intra_left/right` = mean connectivity within each hemisphere, `cross` = inter-hemispheric
3. Z-score with rolling baseline; clip ±3; threshold at zero → LEFT or RIGHT command

**Key properties:** No training required. ImCoh mode is robust to volume conduction. Encodes spatial structure beyond a single electrode pair.

---

### BCI 4 — Graph + ML (`graph_ml/`) *(Novel)*

A supervised graph-based system that addresses non-stationarity by selecting only those graph features that are simultaneously **stable over time** and **discriminable between classes**.

**Feature selection — the novel contribution:**

After extracting PLV connectivity features (upper-triangle edges + node strengths), each feature is scored on two independent axes:

- **Stability** (Coefficient of Variation): `CV = std(feature) / |mean(feature)|`. Low CV → temporally stable feature.
- **Discriminability** (Symmetric KL Divergence): histogram-based `0.5 · (KL(P‖Q) + KL(Q‖P))` between LEFT and RIGHT class distributions. High KL → separable feature.

**Selection rule:** keep features with CV ≤ 30th percentile AND KL ≥ 70th percentile.

This dual-axis filter is motivated by the hypothesis that non-stationarity preferentially corrupts features that are already unstable, while stable-but-discriminable features are more likely to remain informative across session boundaries.

**Full pipeline:**
1. Bandpass filter EEG (8–30 Hz)
2. Compute PLV matrix (Hilbert transform, 1-second windows)
3. Extract upper-triangle edge weights + node strength features
4. Apply CV + KL feature selection on training data
5. Train ensemble (SVM, Logistic Regression, KNN); select best by validation accuracy
6. Online: selected features → classifier margin → rolling baseline subtraction → binary command

---

## Online Control Loop

All four systems share the same online architecture:

```
LSL Stream (256 Hz, 64 channels)
        │
        ▼
  Preprocessing (bandpass, channel selection)
        │
        ▼
  Feature Extraction (per sliding window)
        │
        ▼
  Baseline Period (first 10 s: collect baseline statistics, no output)
        │
        ▼
  Classification / Thresholding → binary command {LEFT, RIGHT, None}
        │
        ▼
  Game Loop (pygame) → cursor movement → hit / timeout recording
```

**Baseline adaptation:** Each system estimates its own per-session baseline to account for between-session drift. CSP and Graph ML subtract a baseline margin; Neurofeedback and Graph Asymmetry use rolling z-scoring.

---

## Outcome Taxonomy

Each trial is classified into one of five mutually exclusive outcomes:

| Outcome | Definition |
|---------|------------|
| **Hit** | Cursor contacted the correct paddle |
| **Wrong Paddle** | Cursor contacted the incorrect paddle |
| **Timeout Close Strong** | Timed out; cursor passed the ¼-way mark toward target |
| **Timeout Close Weak** | Timed out; cursor on correct side of centre |
| **Timeout Wrong** | Timed out; cursor on wrong side or stationary |

Three accuracy tiers are derived:
- **Hit Rate** = hits / n_trials
- **Broad Accuracy** = (hits + timeout close strong) / n_trials
- **Liberal Accuracy** = (hits + timeout close strong + timeout close weak) / n_trials

A **fighting flag** is also computed per session: raised when the cursor spends < 45% of frames on the correct side AND generates ≥ 15 reversals toward the wrong paddle — indicating that the BCI is actively working against the user's neural signal.

---

## Data Organisation

### Training data

Each BCI module loads training data from prior sessions:

```
<module>/training_results/Subject_<ID>/Session_<N>/session_data.pkl
```

`session_data.pkl` is a dictionary keyed by trial ID:

```python
{
  trial_id: {
      "eeg":      np.ndarray  # shape [64, T] — raw EEG at 256 Hz
      "label":    int          # 0 = LEFT, 1 = RIGHT
      "hit":      bool
      "cursor_x": list         # cursor x-position per game frame
  },
  ...
}
```

### Training source configuration

`config.py` in each module specifies which prior sessions to train on:

```python
SUBJECT_ID         = "006"
CURRENT_SESSION_ID = "003"
TRAIN_APPS         = ["neurofeedback"]   # which module's data to load
TRAIN_MODE         = "last"              # "last" = most recent prior session only
                                         # "all"  = all prior sessions
```

CSP and Graph ML can be trained on data recorded from *any* of the four BCI systems (cross-domain generalisation).

---

## Hardware & Signal Acquisition

- **Amplifier**: g.tec g.Hiamp (64 channels, 256 Hz)
- **Streaming**: Lab Streaming Layer (LSL); MATLAB Simulink bridge (`gtec_to_LSL.slx`) streams EEG to Python
- **Channels**: 64-channel 10-20 layout; motor region subset (FC/C/CP electrodes) used for most decoders
- **Impedance**: checked before each session; values logged in `ImpedanceValues_*.txt`

---

## Running the Systems

### Simulate a session (no hardware)

```bash
# In one terminal — replay a recorded session as an LSL stream
python csp/sim_lsl_mi.py

# In another terminal — run the BCI
python csp/main.py
```

### Live session

```bash
# 1. Start MATLAB gtec_to_LSL.slx to stream from amplifier
# 2. Edit config.py — set SUBJECT_ID, CURRENT_SESSION_ID, TRAIN_MODE
# 3. Run the BCI
python csp/main.py      # or graph_ml/main.py, graph_neurofeedback/main_plv.py, etc.
```

### Post-session analysis

```bash
python global_accuracy_tracks.py
# Outputs: analysis_outputs/all_trials.csv, all_sessions.csv, <game>_*.csv
```

---

## Configuration

Edit `<module>/config.py` before each session. Key parameters:

| Parameter | Description | Typical value |
|-----------|-------------|--------------|
| `SUBJECT_ID` | Subject identifier | `"006"` |
| `CURRENT_SESSION_ID` | Session being recorded | `"003"` |
| `TRAIN_APPS` | Source module(s) for training data | `["neurofeedback"]` |
| `TRAIN_MODE` | `"last"` or `"all"` prior sessions | `"last"` |
| `SAMPLING_RATE` | EEG sample rate (Hz) | `256` |
| `WINDOW_SIZE` | Inference window (samples) | `256` (1 s) or `512` (2 s) |
| `BP_LO`, `BP_HI` | Bandpass limits (Hz) | `8`, `30` |
| `CSP_NCOMP` | Number of CSP components | `6` |
| `CV_KEEP_PCTL` | Stability percentile threshold (Graph ML) | `30` |
| `KL_KEEP_PCTL` | Discriminability percentile threshold (Graph ML) | `70` |
| `PLV_METHOD` | Connectivity method (Graph Asymmetry) | `"imcoh"` or `"plv"` |

---

## Dependencies

```
numpy
scipy
scikit-learn
mne
pylsl
pygame
matplotlib
```

Install with:

```bash
pip install numpy scipy scikit-learn mne pylsl pygame matplotlib
```

---

## Context

This codebase supports PhD research (Year 2) investigating non-stationarity in EEG-based motor imagery BCIs. The central hypothesis is that graph-based representations of neural connectivity are more robust to cross-session distributional shift than spatial-filter or spectral methods — because connectivity patterns encode relative rather than absolute signal properties, and because the novel CV + KL feature selection step explicitly discards features that are unstable over time.
