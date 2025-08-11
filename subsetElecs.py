# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:40:46 2025


This script will compare two lists: 
    1) the current EEG headset electrode number and order
    2) the Stieger dataset electrode number and order
    
We aim to find the number of shared electrodes, and then the indices 
such that we can properly order the electordes in the online study to 
produce the right PLV map. 

@author: uceerjp
"""

# Define your headset's electrode labels in channel order (1-based)
headset_electrodes = [
    'FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
    'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1', 'A2'
]

# Stieger dataset electrode labels (from the coordinates section)
stieger_electrodes = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

# Normalize capitalization
headset_electrodes = [ch.upper() for ch in headset_electrodes]
stieger_electrodes = [ch.upper() for ch in stieger_electrodes]

# Step 1: Check which electrodes from Stieger are present in the headset
present_in_headset = [elec for elec in stieger_electrodes if elec in headset_electrodes]
missing_in_headset = [elec for elec in stieger_electrodes if elec not in headset_electrodes]

# Step 2: Find if the order matches (only for shared electrodes)
shared_order_matches = True
reorder_indices = []

for elec in present_in_headset:
    headset_idx = headset_electrodes.index(elec)
    stieger_idx = stieger_electrodes.index(elec)
    reorder_indices.append(headset_idx)
    if headset_idx != stieger_idx:
        shared_order_matches = False

# Display results
print("✅ Electrodes in both Stieger and Headset ({}):\n{}".format(len(present_in_headset), present_in_headset))
print("\n❌ Electrodes in Stieger but missing in Headset ({}):\n{}".format(len(missing_in_headset), missing_in_headset))

if shared_order_matches:
    print("\n✅ Electrode order matches between datasets.")
else:
    print("\n⚠️ Electrode order does NOT match. Use the following index mapping to reorder your Stieger data:")
    print("Reorder indices (to match headset):\n", reorder_indices)

# Optional: Create a mapping dictionary from Stieger index to headset index
mapping_dict = {stieger_electrodes.index(elec): headset_electrodes.index(elec) for elec in present_in_headset}
print("\n📌 Mapping dict (Stieger idx -> Headset idx):")
print(mapping_dict)
