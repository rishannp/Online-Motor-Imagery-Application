# -*- coding: utf-8 -*-
"""
Replay-recorded online session EEG through LSL.

Goal:
- Load session_data.pkl (your actual online trials)
- Stream samples over LSL as a continuous EEG stream
- So main_online.py sees identical data distribution to future live online

Author: Rishan (edited by ChatGPT)
"""

# =========================
# User-adjustable parameters
# =========================

PKL_PATH = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\Online-Motor-Imagery-Decoder\Experiment Scripts\neurofeedback\training_results\Subject_000\Session_006\session_data.pkl"

STREAM_NAME = "ReplayOnlineSession"
STREAM_TYPE = "EEG"

# If your online code sees nominal_srate=256, keep 256.
NOMINAL_SRATE_HZ = 256

# Playback controls
LOOP_FOREVER = True
PLAYBACK_SPEED = 1.0          # 1.0 = real-time, 2.0 = 2x faster, 0.5 = half speed
START_TRIAL_INDEX = 0         # start replay from this trial
MAX_TRIALS = None             # None = all trials
INTER_TRIAL_GAP_S = 0.0       # optional pause between trials (seconds)

# Data interpretation controls
# Many pipelines store EEG in Volts; some store in microvolts.
# If your session_data.pkl stores microvolts, set SCALE_TO_VOLTS = 1e-6
# If it already stores Volts, keep 1.0
SCALE_TO_VOLTS = 1.0

# If recorded trial arrays are shaped (channels, samples) set this True.
# If shaped (samples, channels) set False.
CHANNELS_FIRST = None  # None = auto-detect, otherwise True/False

# If your recording has more/less than 64 channels:
# - If >64 and includes HEADSET_64 order, we can index them.
# - If !=64 and we can't reconcile, we hard error.
REQUIRE_64CH = True


# =========================
# Implementation
# =========================

import time
import pickle
import numpy as np

from pylsl import StreamInfo, StreamOutlet, local_clock

# Must match your online preprocessing expectations
from preprocess import HEADSET_64


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _infer_orientation(arr: np.ndarray):
    """
    Returns (channels_first: bool).
    Heuristic:
    - if one dimension equals 64, likely channels dimension
    - else choose smaller dim as channels if plausible
    """
    if arr.ndim != 2:
        raise ValueError(f"EEG array must be 2D, got shape={arr.shape}")

    a, b = arr.shape

    if a == 64 and b != 64:
        return True
    if b == 64 and a != 64:
        return False

    # If neither dimension is 64, guess: channels usually smaller than samples
    return (a < b)


def _extract_trials(obj):
    """
    Try to extract a list of trials where each trial is a 2D array:
    either [channels, samples] or [samples, channels].

    Supports common formats:
    - list of dicts with keys: 'eeg', 'EEG', 'data', 'x', 'window', 'samples'
    - dict with key 'trials' or 'data'
    - dict of arrays
    """
    # Case 1: list
    if isinstance(obj, list):
        trials = []
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                for k in ("eeg", "EEG", "data", "x", "window", "samples"):
                    if k in item:
                        trials.append(_as_numpy(item[k]))
                        break
                else:
                    # dict but no known key
                    raise KeyError(
                        f"Trial {i} is dict but contains none of expected keys "
                        f"('eeg','EEG','data','x','window','samples'). Keys={list(item.keys())}"
                    )
            else:
                # list of arrays
                trials.append(_as_numpy(item))
        return trials

    # Case 2: dict with 'trials' or 'data'
    if isinstance(obj, dict):
        if "trials" in obj:
            return _extract_trials(obj["trials"])
        if "data" in obj:
            return _extract_trials(obj["data"])

        # Sometimes stored as dict with numeric keys or similar
        # If values look like arrays/dicts, attempt
        vals = list(obj.values())
        if len(vals) > 0:
            return _extract_trials(vals)

    raise TypeError(f"Unsupported session_data.pkl format: type={type(obj)}")


def _validate_and_fix_trial(trial: np.ndarray):
    """
    Ensure trial is float32 and oriented as [samples, channels] for streaming.
    """
    if trial is None:
        raise ValueError("Trial is None")

    trial = _as_numpy(trial)

    if trial.ndim != 2:
        raise ValueError(f"Each trial must be 2D array, got shape={trial.shape}")

    # Infer orientation if needed
    ch_first = CHANNELS_FIRST
    if ch_first is None:
        ch_first = _infer_orientation(trial)

    if ch_first:
        # [ch, samples] -> [samples, ch]
        trial = trial.T

    # Now trial is [samples, channels]
    n_samples, n_ch = trial.shape

    if REQUIRE_64CH and n_ch != 64:
        raise ValueError(
            f"Expected 64 channels after orientation fix, got {n_ch}. "
            f"Trial shape (samples, ch)={trial.shape}. "
            f"If your data truly isn't 64ch, set REQUIRE_64CH=False and adjust pipeline."
        )

    # Convert to volts scaling if needed
    trial = trial.astype(np.float32) * float(SCALE_TO_VOLTS)

    return trial


def _make_outlet():
    info = StreamInfo(
        name=STREAM_NAME,
        type=STREAM_TYPE,
        channel_count=64,
        nominal_srate=float(NOMINAL_SRATE_HZ),
        channel_format="float32",
        source_id=f"{STREAM_NAME}_source"
    )

    # Add channel labels (helps debugging & consistency)
    chns = info.desc().append_child("channels")
    for label in HEADSET_64:
        ch = chns.append_child("channel")
        ch.append_child_value("label", str(label))
        ch.append_child_value("unit", "volts")
        ch.append_child_value("type", "EEG")

    outlet = StreamOutlet(info, chunk_size=0, max_buffered=360)
    return outlet


def main():
    # Load pkl
    with open(PKL_PATH, "rb") as f:
        session_obj = pickle.load(f)

    trials_raw = _extract_trials(session_obj)
    if len(trials_raw) == 0:
        raise ValueError("No trials found in session_data.pkl")

    # Apply start/max slicing
    end_idx = len(trials_raw) if MAX_TRIALS is None else min(len(trials_raw), START_TRIAL_INDEX + int(MAX_TRIALS))
    trials_raw = trials_raw[START_TRIAL_INDEX:end_idx]

    trials = [_validate_and_fix_trial(t) for t in trials_raw]

    outlet = _make_outlet()

    print(f"Loaded trials: {len(trials)} from:")
    print(PKL_PATH)
    print(f"Streaming as LSL: name='{STREAM_NAME}', type='{STREAM_TYPE}', srate={NOMINAL_SRATE_HZ}Hz, ch=64")
    print(f"PLAYBACK_SPEED={PLAYBACK_SPEED} LOOP_FOREVER={LOOP_FOREVER} SCALE_TO_VOLTS={SCALE_TO_VOLTS}")

    # Timing
    dt = 1.0 / float(NOMINAL_SRATE_HZ)
    dt_play = dt / float(PLAYBACK_SPEED)

    # Replay loop
    trial_idx = 0
    while True:
        if trial_idx >= len(trials):
            if LOOP_FOREVER:
                trial_idx = 0
            else:
                print("Replay finished.")
                break

        trial = trials[trial_idx]  # [samples, 64]
        n_samples = trial.shape[0]

        # Stream sample-by-sample to mimic real acquisition timing
        # (You can speed up by pushing chunks, but timing realism is useful for online pipelines.)
        t0 = local_clock()
        for i in range(n_samples):
            outlet.push_sample(trial[i].tolist())

            # pace
            target = t0 + (i + 1) * dt_play
            now = local_clock()
            sleep_s = target - now
            if sleep_s > 0:
                time.sleep(sleep_s)

        if INTER_TRIAL_GAP_S > 0:
            time.sleep(float(INTER_TRIAL_GAP_S))

        trial_idx += 1


if __name__ == "__main__":
    main()
