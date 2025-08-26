# main_training.py

import json
from queue import Queue
from threading import Thread
from collections import deque
import time

import numpy as np
from config import SESSION_DIR, WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE
from lsl_stream import _inlet
from training_pipeline import ARPipeline
from game import run_game

def main():
    # save config snapshot
    with open(f"{SESSION_DIR}/config.json", 'w') as f:
        json.dump({k:repr(v) for k,v in vars(__import__('config')).items()
                   if k.isupper()}, f, indent=2)

    pipeline   = ARPipeline(band=(10.5, 13.5), order=16)
    action_q   = Queue()          # left/right commands to the game
    adapt_q    = Queue()          # (unused in trainer, kept for API parity)
    label_q    = Queue()          # (unused in trainer, kept for API parity)
    raw_eeg_q  = deque(maxlen=1)  # last WINDOW_SIZE block for any inspection
    eeg_chunk_q = Queue()         # NEW: continuous sample chunks for saving

    def bci_loop():
        """
        - Push EVERY SAMPLE to eeg_chunk_q so the game can build a continuous
          per-trial buffer with no overlap or gaps.
        - Every STEP_SIZE (~40 ms), run AR pipeline on the last WINDOW_SIZE
          samples to emit a control command.
        """
        buf = deque(maxlen=WINDOW_SIZE)
        count = 0

        while True:
            sample, _ = _inlet.pull_sample(timeout=1.0 / SAMPLING_RATE)
            if not sample:
                continue

            # Stream every sample to the game for continuous recording
            eeg_chunk_q.put(np.asarray(sample, dtype=float)[np.newaxis, :])  # [1, n_ch]

            # Maintain sliding window for inference
            buf.append(sample)
            count += 1

            # Run AR pipeline ~every 40 ms once we have a full window
            if len(buf) == WINDOW_SIZE and count % STEP_SIZE == 0:
                window = np.array(buf)  # [WINDOW_SIZE, n_ch]
                raw_eeg_q.clear()
                raw_eeg_q.append(window)

                cmd = pipeline.process(window)  # 0=Left, 1=Right
                action_q.put(cmd)

    Thread(target=bci_loop, daemon=True).start()

    # Launch the same lane-runner game. We pass the NEW eeg_chunk_q so it can
    # build a continuous buffer per trial.
    run_game(action_q, adapt_q, [], label_q, raw_eeg_q, eeg_chunk_q)

if __name__ == '__main__':
    main()
