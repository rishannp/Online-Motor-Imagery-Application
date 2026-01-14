# sim_lsl_mi.py
# I use this to simulate an LSL EEG stream for MI testing.
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np, time
from preprocess import Preprocessor

fs = 256
pp = Preprocessor()
labels = pp.headset_electrodes  # 64 labels in your known order

info = StreamInfo('SimMI', 'EEG', len(labels), fs, 'float32', 'SIM001')
chns = info.desc().append_child("channels")
for lab in labels:
    ch = chns.append_child("channel")
    ch.append_child_value("label", lab)
outlet = StreamOutlet(info)

# indices
idx = {lab: i for i, lab in enumerate(labels)}
C3, FC3, CP3 = idx['C3'], idx['FC3'], idx['CP3']
C4, FC4, CP4 = idx['C4'], idx['FC4'], idx['CP4']

t = 0
phase = 0.0

REST_SECONDS = 10.0      # I force an initial dead rest block for baseline
block_len = 10.0         # seconds per MI condition
session_start = local_clock()

print("Publishing SimMI LSL stream... (Ctrl+C to stop)")
print(f"Mode: REST {REST_SECONDS:.0f}s, then alternating LEFT/RIGHT every {block_len:.0f}s")

try:
    while True:
        now = local_clock()
        elapsed = now - session_start

        # determine condition: rest first, then alternate left/right
        if elapsed < REST_SECONDS:
            cond = -1  # REST
        else:
            cond = int(((elapsed - REST_SECONDS) // block_len) % 2)  # 0=Left, 1=Right

        # base noise
        sample = 5e-6 * np.random.randn(len(labels))  # ~5 uV noise

        # 12 Hz alpha carrier
        alpha = np.sin(2*np.pi*12*(t/fs) + phase)

        if cond == -1:
            # REST: symmetric alpha so baseline is unbiased
            sample[C4] += 2e-6 * alpha
            sample[C3] += 2e-6 * alpha
        else:
            # Lateralize: decrease alpha on contralateral side (desync)
            # Left imagery (cond=0): reduce C4; Right imagery (cond=1): reduce C3
            if cond == 0:
                sample[C4] += 2e-6 * alpha * 0.3
                sample[C3] += 2e-6 * alpha * 1.2
            else:
                sample[C4] += 2e-6 * alpha * 1.2
                sample[C3] += 2e-6 * alpha * 0.3

        # neighbors get smaller amplitude (kept simple on purpose)
        for ch in (FC3, CP3, FC4, CP4):
            sample[ch] += 1e-6 * alpha

        outlet.push_sample(sample.tolist())
        t += 1
        time.sleep(1.0/fs)

except KeyboardInterrupt:
    print("\nStopped.")
