# 🧠 BCI Lane Runner (Adaptive Online EEG Control)

Welcome to **BCI Lane Runner**, an adaptive Brain-Computer Interface (BCI) game designed for **real-time online EEG decoding** using CSP or GAT-based classifiers. The player moves left or right to intercept falling targets, controlled entirely by your brain signals (or keyboard fallback for testing). Built for **online adaptation**, model retraining happens seamlessly between levels using correctly classified EEG windows.

It is fair to operate as though this work is to reproduce, and build upon the state-of-the-art literature published here: 
- https://ieeexplore.ieee.org/abstract/document/1634519
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123727
- https://ieeexplore.ieee.org/abstract/document/4015588
- https://ieeexplore.ieee.org/abstract/document/6177271
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0076214

---

## 🏑 Overview

- Real-time BCI loop using EEG data via Lab Streaming Layer (LSL)
- Feature extraction via **PLV + Graph Attention Network (GAT)** or **CSP + LDA**
- Online adaptation every *N* correct trials
- 2-lane runner-style game with balanced left/right class presentation
- Full trial logging: EEG windows, spawn/goal positions, outcomes

---

## 🧩 System Architecture

```
[LSL Stream] → [Preprocessing] → [Classifier] → [Prediction Queue]
                                            ↘ [Correct Trials] → [Adaptation]
[Game Engine (Pygame)] ← [Action Queue] ←────
                            ↑
                  [EEG Snapshot per Frame]
```

---

## ⚙️ Configuration

All settings live in `config.py`:

```python
# Static vs Adaptive
ADAPTATION      = True       # toggle adaptation
ADAPT_N         = 10         # N correct trials before retraining

# Game settings
NUM_LEVELS        = 10       # number of levels in total
TRIALS_PER_LEVEL  = 20       # number of trials per level

# EEG Pipeline
WINDOW_SIZE     = 256 * 3    # 3-second windows at 256Hz
STEP_SIZE       = 128        # 75% overlap
BUFFER_SIZE     = 10         # smoothing buffer for prediction
THRESHOLD       = 3          # majority-vote threshold
SAMPLING_RATE   = 256        # Hz

# Paths to pretrained model
TRAINING_DATA   = "./training_data.pkl"
GAT_MODEL_PT    = "./best_finetuned_model.pt"

# Subject/session
SUBJECT_ID      = "001"
RESULTS_DIR     = "./results"
```

---

## 🚀 Running the Experiment

1. Start your EEG acquisition software and ensure an LSL stream is available.
2. Activate your Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the experiment:
   ```bash
   python main_online.py
   ```

Keyboard fallback: Use the ← / → keys if no EEG is available or for debugging.

---

## 🎮 Gameplay Description

- The screen shows a player circle at the bottom and a falling goal block.
- At each trial, the block spawns on the **left or right lane** (random but balanced).
- Move your cursor (via BCI prediction) to intercept the block = **hit**, else **miss**.
- Each **level contains N trials** (hits + misses). After N trials:
  - Model retrains using **correctly classified trials**.
  - "Adapting" flashes in **green** for 1 second as a visual cue.
- Game continues through `NUM_LEVELS`.

---

## 🧠 Model Training & Adaptation

- Two modes:
  - `method='plv'`: Uses Phase-Locking Value (PLV) matrix → GATv2
  - `method='csp'`: Uses CSP + Linear Discriminant Analysis
- Online adaptation:
  - Only trials where the classifier's prediction matches the true label are stored.
  - After `ADAPT_N` correct trials are collected, the model is fine-tuned.

---

## 📦 Output Format

After each session, results are saved to:

```
./results/Subject<id>_Session.pkl
```

Each session is a list of `trial` dictionaries:

```python
{
  'spawn_timestamp': 123456,
  'spawn_player_x': 200,
  'spawn_goal_x': 50,
  'label': 0,  # 0 = left, 1 = right
  'outcome_timestamp': 123789,
  'end_player_x': 180,
  'end_goal_x': 48,
  'outcome': 'hit',  # or 'miss'
  'eeg_windows': [np.array(...), ...]  # preprocessed EEG windows
}
```

Frame-by-frame game logs are also accessible in `game_states` (in-memory list).

---

## 📊 Analyzing Data

Load the session data like this:

```python
import pickle
from config import SUBJECT_ID, RESULTS_DIR
import os

path = os.path.join(RESULTS_DIR, f"Subject{SUBJECT_ID}_Session.pkl")
with open(path, 'rb') as f:
    trials = pickle.load(f)

# Example: count hits
num_hits = sum(t['outcome'] == 'hit' for t in trials)
```

---

## 🔧 Extend & Customize

Ideas for customization:

- Include other models!
- Add a third rest state class
- Visualize model accuracy in real time
- Add calibration or scoring screens between levels

---

## 🤝 Contributing

Contributions welcome! You can:

- Improve EEG preprocessing (e.g., ICA, ASR)
- Add new feature extraction methods
- Optimize adaptation routines
- Improve UI/UX for participants

Fork the repo and submit a pull request with a clear description of your changes.

---

## 📜 License

MIT License. See `LICENSE.md`.

---

## 👤 Author

Developed by [Rishan Patel](https://github.com/rishanp), UCL PhD Candidate in Brain-Computer Interfaces.

---

## 💡 Citation

If this tool helps your research, feel free to cite or acknowledge the repository. Future publication incoming.

```
```
