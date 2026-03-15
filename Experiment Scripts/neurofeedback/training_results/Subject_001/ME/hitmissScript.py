# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:43:50 2025
@author: uceerjp
"""

# quick_hits_misses.py
import sys
import os
import pickle

def iter_trials(data_obj):
    """Yield trial dicts from either a dict or list structure."""
    if isinstance(data_obj, dict):
        # assume dict of {trial_id: trial_dict}
        for _, tr in sorted(data_obj.items(), key=lambda kv: kv[0]):
            yield tr
    elif isinstance(data_obj, list):
        for tr in data_obj:
            yield tr
    else:
        raise TypeError(f"Unsupported data container: {type(data_obj)}")

def main(path="session_data.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find {path}. Pass a different path as an argument.")

    with open(path, "rb") as f:
        data = pickle.load(f)

    hits = 0
    misses = 0
    total = 0

    for tr in iter_trials(data):
        # default to False if missing; cast to bool in case it's 0/1
        is_hit = bool(tr.get("hit", False))
        hits += int(is_hit)
        misses += int(not is_hit)
        total += 1

    hit_rate = (hits / total * 100.0) if total else 0.0

    print(f"File: {path}")
    print(f"Total trials: {total}")
    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    print(f"Hit rate: {hit_rate:.1f}%")

if __name__ == "__main__":
    # Usage: python quick_hits_misses.py [optional_path_to_pkl]
    p = sys.argv[1] if len(sys.argv) > 1 else "session_data.pkl"
    main(p)
