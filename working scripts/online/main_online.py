# Online loop that emits ~every 40 ms (with processing delay) after an initial 3 s warm-up,
# using the *actual* LSL stream rate to compute window + hop sizes.
import json
import time
import threading
from collections import deque
from queue import Queue
from threading import Thread

import numpy as np
from config import (
    METHOD, SUBJECT_DIR, SESSION_DIR,
    VISUALISE_PLV, WINDOW_SIZE, FEEDBACK_INTERVAL,
    SAMPLING_RATE
)
from preprocess import preprocess_window
from featandclass import BCIPipeline, n_channels
from lsl_stream import _inlet, shutdown_lsl
from game import run_game

def main():
    # 1) Write out config snapshot
    with open(f"{SESSION_DIR}/config.json", 'w') as j:
        json.dump({k: repr(v) for k, v in vars(__import__('config')).items()
                   if k.isupper()}, j, indent=2)

    # 2) Initialize BCI pipeline + queues + stop_event
    pipeline     = BCIPipeline(method=METHOD)
    action_queue = Queue()
    label_queue  = Queue()
    adapt_queue  = Queue()
    raw_eeg_log  = deque(maxlen=1)
    stop_event   = threading.Event()

    # 3) Optional PLV visualiser
    def plv_visualiser():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(np.zeros((n_channels, n_channels)), cmap='hot', vmin=0, vmax=1)
        ax.set_title("Live PLV")
        plt.colorbar(im, ax=ax)
        plt.ion(); plt.show()
        last = None
        while not stop_event.is_set():
            plv = pipeline.latest_plv
            if plv is not None and not np.array_equal(plv, last):
                im.set_data(plv)
                last = plv.copy()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            time.sleep(0.1)

    # 4) BCI loop — uses runtime stream rate for exact hop timing
    def bci_loop():
        try:
            sr_stream = int(_inlet.info().nominal_srate()) or int(SAMPLING_RATE)
        except Exception:
            sr_stream = int(SAMPLING_RATE)

        # Interpret WINDOW_SIZE as "samples @ SAMPLING_RATE"; convert to stream samples
        window_sec  = WINDOW_SIZE / float(SAMPLING_RATE)
        win_samples = max(1, int(round(window_sec * sr_stream)))
        hop_samples = max(1, int(round(FEEDBACK_INTERVAL * sr_stream)))

        print(f"[BCI] stream_sr={sr_stream} Hz | window={win_samples} samples (~{window_sec:.2f}s) "
              f"| hop={hop_samples} samples (~{hop_samples/sr_stream:.3f}s)")

        buf = deque(maxlen=win_samples)
        total_samples = 0
        next_emit_at = None
        last_emit_ts = None

        while not stop_event.is_set():
            sample, ts = _inlet.pull_sample(timeout=1.0 / max(1, sr_stream))
            if not sample:
                continue

            buf.append(sample)
            total_samples += 1

            if len(buf) < win_samples:
                continue

            if next_emit_at is None:
                next_emit_at = total_samples  # emit immediately at first full window

            if total_samples >= next_emit_at:
                next_emit_at += hop_samples

                window = np.array(buf)           # [win_samples, n_channels]
                proc   = preprocess_window(window)
                if proc is None:
                    continue

                raw_eeg_log.clear()
                raw_eeg_log.append(proc)

                cmd = pipeline.predict(proc)
                now = time.time()
                if last_emit_ts is not None:
                    print(f"[{time.strftime('%H:%M:%S')}] Prediction: {cmd} "
                          f"(Δ={now - last_emit_ts:.3f}s)")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Prediction: {cmd} (first)")
                last_emit_ts = now

                action_queue.put(cmd)

                # Adaptation intake
                if pipeline.adaptive and not label_queue.empty():
                    lbl, windows = label_queue.get()
                    for w in windows:
                        pipeline._win_buf.append(w)
                        pipeline._lab_buf.append(lbl)
                    if len(pipeline._win_buf) >= pipeline.adapt_N:
                        t0 = time.perf_counter()
                        pipeline.adapt()
                        t1 = time.perf_counter()
                        adapt_queue.put(int((t1 - t0) * 1000))

        # Graceful shutdown of LSL
        shutdown_lsl()

    # Start threads (avoid duplicates if rerun)
    names = {t.name for t in threading.enumerate()}
    if "bci_loop" not in names:
        Thread(target=bci_loop, name="bci_loop", daemon=True).start()
    else:
        print("[BCI] Loop already running; not starting another.")

    if VISUALISE_PLV and METHOD.lower() == 'plv' and "plv_vis" not in names:
        Thread(target=plv_visualiser, name="plv_vis", daemon=True).start()

    # 5) Launch game (blocks). Press K/Esc or close window to quit.
    run_game(action_queue, adapt_queue, [], label_queue, raw_eeg_log, stop_event)

    # Ensure shutdown after game exits (also covers K/Esc)
    stop_event.set()
    shutdown_lsl()
    time.sleep(0.1)

if __name__ == '__main__':
    main()
