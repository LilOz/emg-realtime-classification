import threading
import time
from collections import deque
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from pyOpenBCI import OpenBCICyton
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

# Feature extraction (can still use your own module)
from feature_extractors import iemg, ln_rms, var, msv

POSE_MAP = {
    1: "rest",
    2: "fist",
    3: "wrist flexion (down)",
    4: "wrist extension (up)",
    5: "radial deviation (right)",
    6: "ulnar deviation (left)",
}
SCALE_FACTOR_EEG = (4500000) / 24 / (2**23 - 1)  # uV/count

# ====== Parameters ======
fs = 250
window_duration = 0.5
buffer_size = int(fs * window_duration)
time_step = 1 / fs
plot_buffer_len = 250  # 1s of data per channel

lowcut, highcut, notch_freq, order = 20, 124, 50, 8
nyq = 0.5 * fs
b_band, a_band = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
b_notch, a_notch = iirnotch(notch_freq / nyq, 30)

# ====== Shared Buffers ======
channel_buffers = [deque(maxlen=buffer_size) for _ in range(8)]  # for classification
plot_buffers = [deque(maxlen=plot_buffer_len) for _ in range(8)] # for plotting

band_states = [lfilter_zi(b_band, a_band) * 0 for _ in range(8)]
notch_states = [lfilter_zi(b_notch, a_notch) * 0 for _ in range(8)]

# ====== Load Models ======
model = joblib.load("emg_rf_model.pkl")
scaler = joblib.load("emg_scaler.pkl")


# ====== Feature Extraction ======
def extract_features(window, feature_extractors=[iemg, msv, var, ln_rms]):
    features = []
    channel_data = []

    for i in range(8):
        x = window[f" EXG Channel {i}"].values
        channel_data.append(x)
        for extractor in feature_extractors:
            result = extractor(x)
            features.extend(
                result if isinstance(result, (list, np.ndarray)) else [result]
            )

    channel_data = np.array(channel_data)
    for i, j in combinations(range(8), 2):
        features.append(np.corrcoef(channel_data[i], channel_data[j])[0, 1])

    return features


# ====== Classification ======
def classify_real_time(window):
    features = extract_features(window)
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]


# ====== Streaming Thread ======
def stream_thread():
    global band_states, notch_states, last_print_time

    def handle_stream(sample):
        global band_states, notch_states, last_print_time

        for i in range(8):
            x = sample.channels_data[i] * SCALE_FACTOR_EEG
            x_band, band_states[i] = lfilter(b_band, a_band, [x], zi=band_states[i])
            x_notch, notch_states[i] = lfilter(
                b_notch, a_notch, x_band, zi=notch_states[i]
            )
            x_notch[0] = np.clip(x_notch[0], -600, 600)
            channel_buffers[i].append(x_notch[0])
            plot_buffers[i].append(x_notch[0])

        if all(len(buf) == buffer_size for buf in channel_buffers):
            if time.time() - last_print_time > window_duration:
                window = pd.DataFrame(
                    {f" EXG Channel {i}": list(channel_buffers[i]) for i in range(8)}
                )
                prediction = classify_real_time(window)
                print("Predicted class:", POSE_MAP.get(prediction, prediction))
                last_print_time = time.time()

    last_print_time = 0

    board = OpenBCICyton(port="/dev/tty.usbserial-DM03H689")
    for ch in range(1, 9):
        cmd = f"x{ch}0400000X"
        board.write_command(cmd)
    print("SRB2 disabled on all 8 channels.")

    print("Starting stream...")
    board.start_stream(handle_stream)


# ====== Start Thread ======
threading.Thread(target=stream_thread, daemon=True).start()

# ====== Plotting ======
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)
lines = []

for i, ax in enumerate(axes):
    line, = ax.plot([], [], label=f"Channel {i+1}")
    lines.append(line)
    ax.set_ylim(-300, 300)  # Adjust based on expected signal range
    ax.set_xlim(0, plot_buffer_len)
    ax.set_ylabel(f"Ch {i+1}")
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Samples")
plt.tight_layout()


def animate(i):
    for idx, line in enumerate(lines):
        y = list(plot_buffers[idx])
        x = list(range(len(y)))
        line.set_data(x, y)
    return lines


ani = FuncAnimation(fig, animate, interval=50, blit=True)
plt.tight_layout()
plt.show()
