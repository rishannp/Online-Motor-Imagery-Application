# main.py  (GRAPH+ML APP ENTRYPOINT)
import json
from queue import Queue
from threading import Thread
from collections import deque

import numpy as np

from config import (
    CURRENT_SESSION_DIR, MODEL_OUT_DIR, TRAIN_SESSION_PKL,
    WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE,
)

from lsl_stream import get_inlet
from graph_training import train_and_save
from graph_pipeline import GraphMLPipeline
from game import run_game

def main():
    # Snapshot config for this run
    with open(f"{CURRENT_SESSION_DIR}/config.json", "w") as f:
        json.dump(
            {k: repr(v) for k, v in vars(__import__("config")).items() if k.isupper()},
            f,
            indent=2,
        )

    print(f"[GRAPH APP] training source pkl: {TRAIN_SESSION_PKL}")
    model_path, pack = train_and_save()
    print(f"[GRAPH APP] saved model pack: {model_path}")
    print(f"[GRAPH APP] model outputs dir: {MODEL_OUT_DIR}")
    print(f"[GRAPH APP] current session dir: {CURRENT_SESSION_DIR}")

    pipeline = GraphMLPipeline(pack)

    action_q    = Queue()
    adapt_q     = Queue()
    label_q     = Queue()
    raw_eeg_q   = deque(maxlen=1)
    eeg_chunk_q = Queue()

    inlet = get_inlet()

    def bci_loop():
        buf = deque(maxlen=WINDOW_SIZE)
        count = 0

        while True:
            sample, _ = inlet.pull_sample(timeout=1.0 / SAMPLING_RATE)
            if not sample:
                continue

            eeg_chunk_q.put(np.asarray(sample, dtype=float)[np.newaxis, :])
            buf.append(sample)
            count += 1

            if len(buf) == WINDOW_SIZE and (count % STEP_SIZE == 0):
                window = np.asarray(buf, dtype=np.float32)  # [WINDOW_SIZE, 64]
                raw_eeg_q.clear()
                raw_eeg_q.append(window)

                cmd = pipeline.process(window)

                # ~1 Hz log
                if count % SAMPLING_RATE == 0:
                    print(f"[GRAPH] cmd={cmd}")

                action_q.put(cmd)

    Thread(target=bci_loop, daemon=True).start()
    run_game(action_q, adapt_q, [], label_q, raw_eeg_q, eeg_chunk_q)

if __name__ == "__main__":
    main()
