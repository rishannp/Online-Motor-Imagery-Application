# main.py  (GRAPH+ML APP)
import json
import time
from queue import Queue
from threading import Thread
from collections import deque

import numpy as np

from config import (
    CURRENT_SESSION_DIR, MODEL_OUT_DIR, TRAIN_SESSION_PKLS,
    WINDOW_SIZE, STEP_SIZE,
)

from lsl_stream import stream_chunk
from graph_training import train_and_save
from graph_pipeline import GraphMLPipeline
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

    print(f"[GRAPH APP] training sources: {[(a, s) for a, s, _ in TRAIN_SESSION_PKLS]}", flush=True)
    model_path, pack = train_and_save()
    print(f"[GRAPH APP] saved model pack: {model_path}", flush=True)
    print(f"[GRAPH APP] model outputs dir: {MODEL_OUT_DIR}", flush=True)
    print(f"[GRAPH APP] current session dir: {CURRENT_SESSION_DIR}", flush=True)

    pipeline = GraphMLPipeline(pack)
    reset_preprocess_state()

    action_q = Queue()
    adapt_q = Queue()
    label_q = Queue()
    raw_eeg_q = deque(maxlen=1)
    eeg_chunk_q = Queue()

    def bci_loop():
        print("[GRAPH] BCI loop started (chunked sliding inference)", flush=True)

        win = int(WINDOW_SIZE)
        step = int(STEP_SIZE)
        buf = deque(maxlen=win)
        samples_since_last_infer = 0
        last_cmd = None
        last_dbg = time.time()

        while True:
            chunk = stream_chunk(max_wait_sec=10.0)
            if chunk is None:
                continue

            eeg_chunk_q.put(chunk)

            for row in chunk:
                buf.append(row)

            samples_since_last_infer += int(chunk.shape[0])

            if len(buf) < win:
                continue
            if samples_since_last_infer < step:
                continue

            n_infers = samples_since_last_infer // step
            samples_since_last_infer = samples_since_last_infer % step

            cmd = last_cmd
            for _ in range(int(n_infers)):
                window = np.asarray(buf, dtype=np.float32)
                raw_eeg_q.clear()
                raw_eeg_q.append(window)
                cmd = pipeline.process(window, n_new=step)

            if cmd is not None:
                action_q.put(cmd)
            last_cmd = cmd

            now = time.time()
            if now - last_dbg > 1.0:
                print(
                    f"[GRAPH] cmd={cmd} chunk={tuple(chunk.shape)} window={tuple(window.shape)} infers={int(n_infers)}",
                    flush=True,
                )
                last_dbg = now

    Thread(target=bci_loop, daemon=True).start()
    run_game(action_q, adapt_q, [], label_q, raw_eeg_q, eeg_chunk_q)


if __name__ == "__main__":
    main()