# Neurofeedback & Online PLV-GAT BCI

This repo contains two related applications:

1. **Neurofeedback (Stieger-style replication)** — a lane‑runner game driven by an online alpha-band AR pipeline. This is used to collect an initial finetuning dataset while presenting stimuli similar to the foundational model’s training setup.
2. **Online PLV + GAT Adaptation** — the same game driven by your foundational GAT model with **adaptive fine‑tuning** over time in the PLV space.

> If you use this code in publications, please **cite our work** (placeholder citation below).

---

## Folder layout

```
neurofeedback/
  ├── config.py             # Global parameters (subject/session, timing, etc.)
  ├── game.py               # Lane‑runner game & trial saving
  ├── preprocess.py         # Channel sets, artifact checks, subsets
  ├── lsl_stream.py         # Connects to an EEG LSL stream
  ├── sim_lsl_mi.py         # Optional LSL simulator for quick testing
  ├── training_pipeline.py  # AR(16) alpha‑band lateralization pipeline
  └── main_training.py      # Entry point for the neurofeedback trainer
  └── trained_models/       # (optional) foundational model weights & trainer scripts

online/
  ├── config.py             # Same interface; add adaptation toggles & paths
  ├── main_online.py        # Entry point for online PLV/GAT with adaptation
  ├── ...                   # Model code, PLV computation, adapters, utils
  └── trained_models/       # Foundational GAT weights (used for warm‑start)
```

---

## Quick start (Neurofeedback)

### 1) Environment
Create and activate a clean Python environment (Python 3.9+ recommended):
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate
```

Install dependencies for the **neurofeedback** app:
```bash
pip install -r requirements-neurofeedback.txt
```

> If you later work in the **online** GAT app, install its requirements too (file to be finalized when those files are added).

### 2) EEG input via LSL
You need an EEG device capable of streaming **LSL** (Lab Streaming Layer).  
For testing without hardware, you can run the simulator:

```bash
python neurofeedback/sim_lsl_mi.py
```

This publishes an LSL stream named `SimMI` with 64 channels in the expected order.

### 3) Configure
Edit `neurofeedback/config.py` to set subject/session, trial counts, timing, etc. The most relevant fields:

- `SUBJECT_ID`, `SESSION_ID`, `RESULTS_DIR` → where outputs are written.
- `NUM_LEVELS`, `TRIALS_PER_LEVEL` → game structure.
- `SAMPLING_RATE`, `WINDOW_SIZE`, `STEP_SIZE` → online processing cadence.
- `METHOD` → fixed to `'ar'` for the neurofeedback trainer.

### 4) Run
Start your LSL source (device or simulator), then:

```bash
cd neurofeedback
python lsl_stream.py        # optional: just to verify connection (prompts to choose a stream)
python main_training.py     # launches BCI loop + game
```

While running, the system computes AR‑based alpha power lateralization (C3 vs C4 small‑Laplacian) every ~40 ms and moves the cursor left/right accordingly. Trials are presented exactly as in the Stieger‑style trainer.

---

## Outputs

- A session snapshot is saved to:
  ```
  {RESULTS_DIR}/Subject_{SUBJECT_ID}/Session_{SESSION_ID}/
  ```
- Files include:
  - `config.json` — the run configuration.
  - `session_data.pkl` — a dict of trials:
    ```python
    trials[trial_id] = {
        'eeg': np.ndarray [n_channels, n_samples],   # continuous EEG for that trial
        'fs': int,                                   # sampling rate (Hz)
        'label': int,                                # 0=Left target, 1=Right target
        'cursor_x': np.ndarray [n_frames],           # cursor x position over time
        'hit': bool
    }
    ```
  - Additional model weights/logs (if produced by other scripts) are stored under the same session directory.

This subject/session‑centric structure is identical in the **online** app so that finetuned models and session data are neatly grouped.

---

## Online PLV + GAT Adaptation (overview)

The `online/` application runs the **same game** but uses a pre‑trained **foundational GAT** model, computes PLV features online, and **adaptively fine‑tunes** the model over time. The user workflow mirrors the trainer:

1. Adjust `online/config.py` (paths to foundational weights, adaptation toggles, subject/session IDs).
2. Start an LSL EEG stream (or a simulator compatible with 64‑ch labels).
3. Run:
   ```bash
   cd online
   python main_online.py
   ```

> Note: Requirements for the **online** app include deep‑learning packages (e.g., `torch`, potentially `torch_geometric`). Once those files are finalized, install with `pip install -r requirements-online.txt`.

---

## Reproducibility

- All mutable parameters live in the local `config.py` of each app.  
- `main_training.py` stores a frozen copy of the config to the session folder for full reproducibility.

---

## Troubleshooting

- **No LSL streams found**: ensure your device or simulator is running before starting the app. On some systems, firewall rules can block LSL.
- **Pygame window not showing / crashing**: verify GPU drivers on Linux and that `pygame` is installed with `SDL` support.
- **Samplerate mismatch**: the trainer assumes `SAMPLING_RATE=256`. If your device differs, align both the device and `config.py` (and re‑run).

---

## Citing

Please cite **our work** if you use this repository:

> Patel, R., *et al.* (YEAR). **TITLE**. *VENUE*. DOI/URL.

(Replace with the final citation when available.)

---

## License

This project is released under the **GNU General Public License v3.0 (GPL‑3.0)**.  
You may use, modify, and share the code under the terms of the GPL‑3.0; **commercial use is not permitted** beyond what the GPL allows without separate permission from the authors. See the full license text in [`LICENSE`](LICENSE).

---

## Acknowledgements

- The neurofeedback trainer replicates the overall stimulus paradigm used in prior literature to enable fair finetuning data collection; implementation is original and tailored for our PLV/GAT pipeline.
