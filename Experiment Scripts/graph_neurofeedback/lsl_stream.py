# lsl_stream.py

import numpy as np
import pylsl
from config import WINDOW_SIZE, SAMPLING_RATE

_streams = pylsl.resolve_streams()
if not _streams:
    raise RuntimeError("No LSL streams found. Ensure your EEG device is online.")

print("\nAvailable LSL Streams:\n")
for i, s in enumerate(_streams):
    print(f"{i}: {s.name()} ({s.type()})")
idx = int(input("Select stream index: "))
_chosen = _streams[idx]
_inlet  = pylsl.StreamInlet(_chosen)
print(f"Connected to '{_chosen.name()}' at {_inlet.info().nominal_srate():.1f} Hz")

def stream_data():
    """Return exactly WINDOW_SIZE samples as [WINDOW_SIZE, n_channels]."""
    buf = []
    while len(buf) < WINDOW_SIZE:
        sample, _ = _inlet.pull_sample(timeout=1.0/SAMPLING_RATE)
        if sample:
            buf.append(sample)
    return np.array(buf)
