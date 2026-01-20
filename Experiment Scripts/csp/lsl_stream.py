# lsl_stream.py
import time
import numpy as np
import pylsl

from config import WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE

# Resolve at import time (as per your old style)
_streams = pylsl.resolve_streams()
if not _streams:
    raise RuntimeError("No LSL streams found. Ensure your EEG device/outlet is online.")

print("\nAvailable LSL Streams:\n", flush=True)
for i, s in enumerate(_streams):
    try:
        print(f"{i}: {s.name()} ({s.type()}) ch={s.channel_count()} @ {s.nominal_srate():.1f}Hz", flush=True)
    except Exception:
        print(f"{i}: <stream> ", flush=True)

idx = int(input("Select stream index: ").strip())
_chosen = _streams[idx]

_inlet = pylsl.StreamInlet(_chosen, max_buflen=60)

# Try to explicitly open (helps some replay outlets)
try:
    _inlet.open_stream(timeout=2.0)
except Exception:
    pass

try:
    info = _inlet.info()
    print(
        f"Connected to '{info.name()}' type={info.type()} ch={info.channel_count()} @ {info.nominal_srate():.1f} Hz",
        flush=True
    )
except Exception:
    print("Connected to selected stream.", flush=True)


def get_inlet():
    """Optional: expose inlet for debugging if needed."""
    return _inlet


def stream_data(max_wait_sec=5.0):
    """
    Return exactly WINDOW_SIZE samples as [WINDOW_SIZE, n_channels].

    If no samples arrive within max_wait_sec, returns None instead of blocking forever.
    """
    buf = []
    t0 = time.time()
    last_print = t0
    timeout = 0.2  # don't use 1/fs on Windows; too jittery

    while len(buf) < int(WINDOW_SIZE):
        sample, _ = _inlet.pull_sample(timeout=timeout)

        if sample is not None:
            buf.append(sample)
            continue

        now = time.time()
        if now - last_print > 1.0:
            print(f"[LSL] waiting for samples... got {len(buf)}/{int(WINDOW_SIZE)}", flush=True)
            last_print = now

        if now - t0 > float(max_wait_sec):
            print("[LSL] stream_data() timed out (no samples).", flush=True)
            return None

    return np.asarray(buf, dtype=np.float32)


def stream_chunk(max_wait_sec=5.0):
    """Return up to STEP_SIZE samples as a float32 numpy array [n, n_channels].

    Uses inlet.pull_chunk(...), which is the correct primitive for sliding inference.
    If no samples arrive within max_wait_sec, returns None.
    """
    step = int(STEP_SIZE)
    t0 = time.time()
    last_print = t0
    timeout = 0.2  # keep small; Windows scheduling jitter is real

    while True:
        chunk, _ = _inlet.pull_chunk(timeout=timeout, max_samples=step)

        if chunk:
            return np.asarray(chunk, dtype=np.float32)

        now = time.time()
        if now - last_print > 1.0:
            print("[LSL] waiting for chunk...", flush=True)
            last_print = now

        if now - t0 > float(max_wait_sec):
            return None
