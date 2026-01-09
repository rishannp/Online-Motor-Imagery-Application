# main_training.py

# Fix: actually run preprocess_window() on each window before evidence/process.

import os
import json
import time
import queue
from queue import Queue
from threading import Thread
from collections import deque

import numpy as np

from config import SESSION_DIR, WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE
from lsl_stream import _inlet
from training_pipeline import ARPipeline
from preprocess import preprocess_window
from game import run_game


def _snapshot_config(out_dir: str):
    # I save a snapshot of config.py constants so each session is reproducible.
    import config as cfg
    d = {
        k: getattr(cfg, k)
        for k in dir(cfg)
        if k.isupper() and not k.startswith("_")
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def main():
    _snapshot_config(SESSION_DIR)

    pipeline = ARPipeline(band=(10.5, 13.5), order=12)

    action_q    = Queue()          # left/right commands to the game (-1/0/1)
    adapt_q     = Queue()          # unused (kept for parity with other modes)
    label_q     = Queue()          # unused (kept for parity with other modes)
    raw_eeg_q   = deque(maxlen=1)  # last WINDOW_SIZE window (for inspection/debug)
    eeg_chunk_q = Queue()          # continuous sample stream to the game
    game_states = Queue()          # BASELINE_START / BASELINE_END from the game

    def bci_loop():
        """
        I continuously read LSL samples, push every sample to eeg_chunk_q (so the
        game can build per-trial buffers), and every STEP_SIZE I decode one window.

        CRITICAL FIX: I run preprocess_window(window) before pipeline.evidence/process.
        """
        buf = deque(maxlen=int(WINDOW_SIZE))
        step_ctr = 0

        baseline_mode = False
        baseline_ds = []

        # I clear any stale baseline parameters at startup.
        pipeline.reset_baseline()

        while True:
            # --- handle state messages from the game (non-blocking) ---
            try:
                while True:
                    st = game_states.get_nowait()
                    if st == "BASELINE_START":
                        baseline_mode = True
                        baseline_ds = []
                        pipeline.reset_baseline()
                    elif st == "BASELINE_END":
                        baseline_mode = False
                        if len(baseline_ds) >= 5:
                            mu = float(np.mean(baseline_ds))
                            sd = float(np.std(baseline_ds, ddof=1)) if len(baseline_ds) > 1 else 1.0
                            pipeline.set_baseline(mu, sd)
                        else:
                            # If baseline is too short, I leave baseline unset (decoder will abstain more).
                            pipeline.reset_baseline()
            except queue.Empty:
                pass

            # --- pull one sample from LSL ---
            sample, _ts = _inlet.pull_sample(timeout=1.0 / float(SAMPLING_RATE))
            if not sample:
                continue

            s = np.asarray(sample, dtype=float)
            buf.append(s)

            # Push every sample to the game so it can store continuous EEG per trial.
            try:
                eeg_chunk_q.put_nowait(s)
            except queue.Full:
                pass

            if len(buf) < int(WINDOW_SIZE):
                continue

            step_ctr += 1
            if step_ctr < int(STEP_SIZE):
                continue
            step_ctr = 0

            window = np.asarray(buf, dtype=float)  # [WINDOW_SIZE, n_ch]
            raw_eeg_q.append(window)

            # -------------------- PREPROCESSING FIX --------------------
            window_pp = preprocess_window(window)
            if window_pp is None:
                # Artifact rejected → abstain.
                try:
                    action_q.put_nowait(-1)
                except queue.Full:
                    pass
                continue
            # -----------------------------------------------------------

            if baseline_mode:
                # I collect baseline evidence stats on the PREPROCESSED data.
                baseline_ds.append(float(pipeline.evidence(window_pp)))
                try:
                    action_q.put_nowait(-1)
                except queue.Full:
                    pass
            else:
                cmd = int(pipeline.process(window_pp))  # -1/0/1
                try:
                    action_q.put_nowait(cmd)
                except queue.Full:
                    pass

    Thread(target=bci_loop, daemon=True).start()

    # Launch the game (it will send BASELINE_START/END events via game_states)
    run_game(action_q, adapt_q, game_states, label_q, raw_eeg_q, eeg_chunk_q)


if __name__ == "__main__":
    main()
