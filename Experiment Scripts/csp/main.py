# main.py  (CSP APP)
import json
import time
from queue import Queue
from threading import Thread
from collections import deque

import numpy as np

from config import (
    CURRENT_SESSION_DIR, MODEL_OUT_DIR, TRAIN_SESSION_PKL,
    WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE,
)

from lsl_stream import stream_chunk
from csp_training import train_and_save
from csp_pipeline import CSPPipeline
from game import run_game
from preprocess import reset_preprocess_state


def main():
    # Snapshot config for THIS current session
    with open(f"{CURRENT_SESSION_DIR}/config.json", "w") as f:
        json.dump(
            {k: repr(v) for k, v in vars(__import__("config")).items() if k.isupper()},
            f,
            indent=2,
        )

    print(f"[CSP APP] training source pkl: {TRAIN_SESSION_PKL}", flush=True)
    model_path, pack = train_and_save()
    print(f"[CSP APP] saved model pack: {model_path}", flush=True)
    print(f"[CSP APP] model outputs dir: {MODEL_OUT_DIR}", flush=True)
    print(f"[CSP APP] current session dir: {CURRENT_SESSION_DIR}", flush=True)

    pipeline = CSPPipeline(pack)

    # Reset causal filter state once at startup (online runs keep state across windows).
    reset_preprocess_state()

    action_q    = Queue()
    adapt_q     = Queue()
    label_q     = Queue()
    raw_eeg_q   = deque(maxlen=1)
    eeg_chunk_q = Queue()

    def bci_loop():
        print("[BCI] loop started (sliding chunk inference)", flush=True)

        win = int(WINDOW_SIZE)
        buf = deque(maxlen=win)  # ring buffer of last WINDOW_SIZE samples

        last_cmd = None
        last_dbg = time.time()

        while True:
            chunk = stream_chunk(max_wait_sec=10.0)  # [n, nch] or None
            if chunk is None:
                continue

            # Save raw EEG chunks for trial logging (game does np.vstack(list_of_chunks))
            eeg_chunk_q.put(chunk)

            # Append chunk into ring buffer sample-by-sample
            for row in chunk:
                buf.append(row)

            if len(buf) < win:
                continue

            window = np.asarray(buf, dtype=np.float32)  # most recent WINDOW_SIZE

            # Maintain raw window for the game to capture per-trial windows
            raw_eeg_q.clear()
            raw_eeg_q.append(window)

            cmd = pipeline.process(window)

            # Optional but recommended: only push command when it changes
            if cmd != last_cmd:
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
