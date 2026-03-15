# main.py  (CSP APP)
import json
import time
from queue import Queue
from threading import Thread
from collections import deque

import numpy as np

from config import (
    CURRENT_SESSION_DIR, MODEL_OUT_DIR, TRAIN_SESSION_PKLS,
    WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE,
)

from lsl_stream import stream_chunk
from csp_training import train_and_save
from csp_pipeline import CSPPipeline
from game import run_game
from preprocess import reset_preprocess_state


def main():
    # Snapshot config for this run
    with open(f"{CURRENT_SESSION_DIR}/config.json", "w") as f:
        json.dump(
            {k: repr(v) for k, v in vars(__import__("config")).items() if k.isupper()},
            f,
            indent=2,
        )

    print(f"[CSP APP] training sources: {[(a, s) for a, s, _ in TRAIN_SESSION_PKLS]}", flush=True)
    model_path, pack = train_and_save()
    print(f"[CSP APP] saved model pack: {model_path}", flush=True)
    print(f"[CSP APP] model outputs dir: {MODEL_OUT_DIR}", flush=True)
    print(f"[CSP APP] current session dir: {CURRENT_SESSION_DIR}", flush=True)

    pipeline = CSPPipeline(pack)
    reset_preprocess_state()

    action_q    = Queue()
    adapt_q     = Queue()
    label_q     = Queue()
    raw_eeg_q   = deque(maxlen=1)
    eeg_chunk_q = Queue()

    def bci_loop():
        print("[BCI] loop started (sliding chunk inference)", flush=True)

        win = int(WINDOW_SIZE)
        buf = deque(maxlen=win)
        last_cmd = None
        last_dbg = time.time()

        while True:
            chunk = stream_chunk(max_wait_sec=10.0)
            if chunk is None:
                continue

            eeg_chunk_q.put(chunk)

            for row in chunk:
                buf.append(row)

            if len(buf) < win:
                continue

            window = np.asarray(buf, dtype=np.float32)
            raw_eeg_q.clear()
            raw_eeg_q.append(window)

            cmd = pipeline.process(window, n_new=chunk.shape[0])

            if cmd is None:
                if last_cmd is not None:
                    action_q.put(None)
                    last_cmd = None
            else:
                cmd = int(cmd)
                if last_cmd is None or cmd != last_cmd:
                    action_q.put(cmd)
                    last_cmd = cmd

            now = time.time()
            if now - last_dbg > 1.0:
                print(
                    f"[BCI] cmd={cmd} chunk_shape={tuple(chunk.shape)} window_shape={tuple(window.shape)}",
                    flush=True,
                )
                last_dbg = now

    Thread(target=bci_loop, daemon=True).start()
    run_game(action_q, adapt_q, [], label_q, raw_eeg_q, eeg_chunk_q)


if __name__ == "__main__":
    main()