import json
import time
from threading import Thread
from collections import deque
from queue import Queue

import numpy as np
import torch

from config import (
    SESSION_DIR, SAMPLING_RATE, WINDOW_SEC, HOP_SEC,
    FOUNDATION_PT, DEVICE_STR
)
from lsl_stream import select_inlet
from preprocess import preprocess_window
from featandclass import GATPredictor
from game import run_game


def main():
    # save config snapshot
    import config as cfg
    with open(f"{SESSION_DIR}/config.json", "w") as f:
        json.dump({k: repr(v) for k, v in vars(cfg).items() if k.isupper()}, f, indent=2)

    inlet = select_inlet()

    predictor = GATPredictor(FOUNDATION_PT, device=DEVICE_STR)

    action_q    = Queue()    # emits 0/1 commands
    eeg_chunk_q = Queue()    # every sample forwarded for continuous recording

    win_samp = int(round(WINDOW_SEC * SAMPLING_RATE))
    hop_samp = max(1, int(round(HOP_SEC * SAMPLING_RATE)))

    def bci_loop():
        buf = deque(maxlen=win_samp)
        count = 0
        last_print = time.time()

        while True:
            sample, _ = inlet.pull_sample(timeout=1.0 / SAMPLING_RATE)
            if not sample:
                continue

            x = np.asarray(sample, dtype=np.float32)
            if x.ndim != 1:
                continue

            # forward every sample to game recording
            eeg_chunk_q.put(x[np.newaxis, :])  # [1, 64]

            # inference buffer
            buf.append(x)
            count += 1

            if len(buf) < win_samp:
                continue

            if count % hop_samp != 0:
                continue

            window_64 = np.stack(buf, axis=0)  # [T, 64]
            window_58 = preprocess_window(window_64, fs=SAMPLING_RATE)
            if window_58 is None:
                continue

            cmd, conf = predictor.predict_lr(window_58)  # 0/1
            action_q.put(cmd)

            if time.time() - last_print > 1.0:
                print(f"cmd={cmd} conf={conf:.3f}")
                last_print = time.time()

    Thread(target=bci_loop, daemon=True).start()
    run_game(action_q, eeg_chunk_q)


if __name__ == "__main__":
    main()
