# main_plv.py

import json
import time
from queue import Queue
from threading import Thread
from collections import deque

import numpy as np

from config import SESSION_DIR, WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE, PLV_METHOD
from lsl_stream import _inlet
from plv_pipeline import PLVAsymmetryPipeline
from game import run_game


def main():
    # Snapshot config for this run
    with open(f"{SESSION_DIR}/config.json", "w") as f:
        json.dump(
            {k: repr(v) for k, v in vars(__import__("config")).items() if k.isupper()},
            f,
            indent=2,
        )

    pipeline = PLVAsymmetryPipeline()
    print(f"[PLV GAME] connectivity method: {pipeline.method.upper()}", flush=True)

    action_q    = Queue()
    adapt_q     = Queue()
    label_q     = Queue()
    raw_eeg_q   = deque(maxlen=1)
    eeg_chunk_q = Queue()

    def bci_loop():
        buf      = deque(maxlen=int(WINDOW_SIZE))
        count    = 0
        last_dbg = time.time()

        while True:
            sample, _ = _inlet.pull_sample(timeout=1.0 / SAMPLING_RATE)
            if not sample:
                continue

            # Every sample goes to game for continuous per-trial EEG recording
            eeg_chunk_q.put(np.asarray(sample, dtype=np.float32)[np.newaxis, :])  # [1, 64]

            buf.append(sample)
            count += 1

            if len(buf) < int(WINDOW_SIZE) or count % int(STEP_SIZE) != 0:
                continue

            window = np.asarray(buf, dtype=np.float32)  # [WINDOW_SIZE, 64]
            raw_eeg_q.clear()
            raw_eeg_q.append(window)

            cmd = pipeline.process(window)  # None | 0 | 1
            action_q.put(cmd)

            now = time.time()
            if now - last_dbg >= 1.0:
                print(
                    f"[{pipeline.method.upper()}] cmd={cmd} "
                    f"L={pipeline.last_intra_left:.4f} "
                    f"R={pipeline.last_intra_right:.4f} "
                    f"X={pipeline.last_cross:.4f} "
                    f"asym={pipeline.last_asym:+.4f} "
                    f"z={pipeline.last_z:+.3f}",
                    flush=True,
                )
                last_dbg = now

    Thread(target=bci_loop, daemon=True).start()
    run_game(action_q, adapt_q, [], label_q, raw_eeg_q, eeg_chunk_q)


if __name__ == "__main__":
    main()
