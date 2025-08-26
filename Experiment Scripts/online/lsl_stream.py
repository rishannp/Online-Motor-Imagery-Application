import numpy as np
import pylsl
from config import WINDOW_SIZE

# Resolve all available LSL streams
_streams = pylsl.resolve_streams()
if not _streams:
    raise RuntimeError("No LSL streams found. Ensure your EEG device is online.")

# Prompt user to select the correct EEG stream
print("\nAvailable LSL Streams:\n")
for idx, stream in enumerate(_streams):
    print(f"{idx}: {stream.name()} ({stream.type()})")
selection = int(input("\nSelect LSL stream index: "))
if selection < 0 or selection >= len(_streams):
    raise ValueError("Invalid stream index selected.")

# Connect to the chosen stream
_chosen = _streams[selection]
_inlet = pylsl.StreamInlet(_chosen)
_time_offset = _inlet.time_correction()
_sampling_rate = _inlet.info().nominal_srate()
print(f"Connected to stream '{_chosen.name()}' at {_sampling_rate:.1f} Hz")

def stream_data():
    """
    Pulls exactly WINDOW_SIZE samples from the selected LSL inlet,
    applies the time correction, and returns a NumPy array of shape
    (WINDOW_SIZE, n_channels).
    """
    buf = []
    while len(buf) < WINDOW_SIZE:
        sample, ts = _inlet.pull_sample(timeout=1.0 / _sampling_rate)
        if sample:
            buf.append(sample)
    return np.array(buf)

def shutdown_lsl():
    """Gracefully close the LSL inlet."""
    try:
        _inlet.close_stream()
        print("[LSL] Inlet closed.")
    except Exception:
        pass
