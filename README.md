# Neurofeedback & Online PLV-GAT BCI

This repository provides two related applications for **motor imagery BCI experiments**:

1. **Neurofeedback (Replicating https://doi.org/10.1093/cercor/bhaa234)**  
   - A left-right goal-target game driven by an AR-based alpha-band pipeline.  
   - Used to collect initial finetuning data under the same paradigm as prior work by https://doi.org/10.1093/cercor/bhaa234.  

2. **Online PLV + GAT Adaptation**  
   - The same game, but with **online PLV-based GAT models** (or CSP+LDA).  
   - Supports **incremental adaptation** during gameplay for subject-specific finetuning.  

> If you use this code, please **cite our work**

---

## 📂 Repository Structure

```
neurofeedback/
  config.py
  game.py
  preprocess.py
  lsl_stream.py
  sim_lsl_mi.py
  training_pipeline.py
  main_training.py
  trained_models/         # foundational model weights & training scripts

online/
  config.py
  game.py
  preprocess.py
  lsl_stream.py
  featandclass.py
  control_logic.py
  main_online.py
  trained_models/         # pretrained models (CSP/LDA .pkl, GAT .pt)
```

---

## ⚙️ Installation

1. Create a virtual environment (Python ≥ 3.9 recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

This single `requirements.txt` covers both `neurofeedback/` and `online/`.

---

## 🧠 Running Neurofeedback (Stieger-style)

### Step 1: Start an EEG source
- **Real EEG device** that streams via **LSL** (Lab Streaming Layer).  
- Or use the included simulator:

```bash
python neurofeedback/sim_lsl_mi.py
```

### Step 2: Configure
Edit `neurofeedback/config.py`:
- `SUBJECT_ID`, `SESSION_ID` → output directories  
- `NUM_LEVELS`, `TRIALS_PER_LEVEL` → experiment length  
- `SAMPLING_RATE`, `WINDOW_SIZE`, etc. → processing cadence  

### Step 3: Run
```bash
cd neurofeedback
python main_training.py
```

---

## 🧠 Running Online PLV + GAT Adaptation

### Step 1: Start an EEG source
Same as above (real LSL stream or simulator).

### Step 2: Configure
Edit `online/config.py`:
- `METHOD` → `"plv"` (Graph Attention Network) or `"csp"` (CSP+LDA)  
- `ADAPTATION` → `True` (online finetuning enabled) or `False` (static model)  
- `ADAPT_N` → number of windows required before adaptation  
- `VISUALISE_PLV` → `True` to open a live PLV heatmap window  

### Step 3: Run
```bash
cd online
python main_online.py
```

The game launches with BCI control. A green “Adapting” indicator shows when the model is being finetuned.

---

## 📊 Outputs

Both apps save results under:

```
collection_results/Subject_{ID}/Session_{ID}/
```

Saved files include:
- `config.json` — frozen config snapshot for reproducibility  
- `session_data.pkl` — dict of trials with EEG, labels, cursor trajectory, outcome  
- Model snapshots:
  - CSP/LDA → `csp_lda_finetuned_N.pkl`  
  - GAT → `gat_finetuned_N.pt`  
- `model_log.json` — chronological list of saved models  

---

## 🔧 Notes

- **Adaptation:**  
  - If enabled, adaptation occurs every `ADAPT_N` labeled windows.  
  - Each new adapted model is saved incrementally.  
- **Visualisation:**  
  - If `VISUALISE_PLV=True`, a live PLV heatmap updates during runtime.

---

## 📖 Citing

If you use this repository in academic work, please cite:

> Patel, R., *et al.* (YEAR). **TITLE**. *VENUE*. DOI/URL.  

To be replaced with publication.

---

## 📜 License

This project is licensed under the **GNU General Public License v3 (GPL-3.0)**.  
You may use, modify, and share the code under the terms of the GPL-3.0.  
**Commercial use is not permitted** without explicit permission from the authors.  

See the full license in [`LICENSE`](LICENSE).
