import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyOpenBCI import OpenBCICyton
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from feature_extractors import (arm_coefficients, iemg, kurt, ln_rms, msv, rms,
                                skewness, var)

POSE_MAP = {
    1: "rest",
    2: "fist",
    3: "wrist flexion (down)",
    4: "wrist extension (up)",
    5: "radial deviation (right)",
    6: "ulnar deviation (left)",
}
SCALE_FACTOR_EEG = (4500000) / 24 / (2**23 - 1)  # uV/count


# ======= Load and Train Model Offline =======
def load_data(file_path="labeled_emg_output.txt"):
    return pd.read_csv(file_path, comment="%", skiprows=0)


def process_data(data, notch_freq=50, highcut=124, lowcut=20, fs=250):
    def bandpass_filter(data, lowcut, highcut, fs, order=8):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data)

    def notch_filter(data, notch_freq, fs, q=30):
        nyq = 0.5 * fs
        w0 = notch_freq / nyq
        b, a = iirnotch(w0, q)
        return lfilter(b, a, data)

    filtered = {}
    for i in range(8):
        ch = f" EXG Channel {i}"
        x = notch_filter(bandpass_filter(data[ch], lowcut, highcut, fs), notch_freq, fs)
        filtered[ch] = np.clip(x, -600, 600)
    filtered["class"] = data["class"]
    filtered_df = pd.DataFrame(filtered)

    return filtered_df[filtered_df["class"] != 0]


def extract_features(
    window,
    feature_extractors=[iemg, msv, var, rms, ln_rms, kurt, skewness, arm_coefficients],
):
    features = []
    for i in range(8):
        x = window[f" EXG Channel {i}"].values
        for fx in feature_extractors:
            val = fx(x)
            features.extend(val if isinstance(val, (list, np.ndarray)) else [val])

    return features


def add_noise(x):
    return x + np.random.normal(0, 0.5 * np.std(x), x.shape)


def scale_amplitude(x, min_scale=0.75, max_scale=1.25):
    factor = np.random.uniform(min_scale, max_scale)
    return x * factor


def augment_window(window):
    """Applies augmentations to EMG window"""
    window_aug = window.copy()
    for i in range(8):  # 8 channels
        x = window_aug[f" EXG Channel {i}"].values
        x = add_noise(x)
        x = scale_amplitude(x)
        window_aug[f" EXG Channel {i}"] = x
    return window_aug


def train_model(
    data,
    base_models,  # List of (name, model) tuples
    window_duration=1.0,
    overlap_percentage=0.125,
    sample_rate=250,
    n_augments=0,
):
    window_size = int(window_duration * sample_rate)
    step = int(window_size * (1 - overlap_percentage))
    X, y, augmented_X, augmented_y = [], [], [], []

    for start in range(0, len(data) - window_size, step):
        window = data.iloc[start : start + window_size]
        dominant_class = window["class"].mode()[0]

        features = extract_features(window)
        X.append(features)
        y.append(dominant_class)

        for _ in range(n_augments):
            aug_window = augment_window(window)
            aug_features = extract_features(aug_window)
            augmented_X.append(aug_features)
            augmented_y.append(dominant_class)

    X.extend(augmented_X)
    y.extend(augmented_y)

    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create voting classifier
    ensemble = VotingClassifier(estimators=base_models, voting="soft")
    ensemble.fit(X_scaled, y)

    print(f"Voting Ensemble Accuracy: {ensemble.score(X_scaled, y) * 100:.2f}%")
    return ensemble, scaler


# ======= Real-Time Stream Setup =======

fs = 250
window_duration = 0.5
buffer_size = int(fs * window_duration)
time_step = 1 / fs

lowcut, highcut, notch_freq, order = 20, 124, 50, 4
nyq = 0.5 * fs

# Filters
b_band, a_band = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
b_notch, a_notch = iirnotch(notch_freq / nyq, 30)

# Initial filter states
band_states = [lfilter_zi(b_band, a_band) * 0 for _ in range(8)]
notch_states = [lfilter_zi(b_notch, a_notch) * 0 for _ in range(8)]

# Rolling buffers
channel_buffers = [deque(maxlen=buffer_size) for _ in range(8)]

# Train the model
raw_data = load_data()
filtered_data = process_data(raw_data)

models = [
    ("lda", LDA()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("svm", SVC(kernel="linear", probability=True, random_state=42)),
]

model, scaler = train_model(
    filtered_data,
    base_models=models,
    window_duration=0.5,
    n_augments=2,
)

# ======= Real-Time Plotting Setup =====

# # Create subplots for each channel
# fig, axs = plt.subplots(4, 2, figsize=(10, 6))
# axs = axs.flatten()
#
# # Initialize lines and axes
# xdata = np.linspace(-window_duration, 0, buffer_size)
# plot_lines = []
# for i in range(8):
#     line, = axs[i].plot(xdata, [0]*buffer_size)
#     axs[i].set_ylim(-600, 600)
#     axs[i].set_xlim(-window_duration, 0)
#     axs[i].set_title(f"Channel {i+1}")
#     plot_lines.append(line)
#
# plt.ion()
# plt.tight_layout()
# plt.show()


# Classification handler
def handle_stream(sample):
    global band_states, notch_states, last_print_time

    for i in range(8):
        x = sample.channels_data[i] * SCALE_FACTOR_EEG

        # Apply bandpass
        x_band, band_states[i] = lfilter(b_band, a_band, [x], zi=band_states[i])
        # Apply notch
        x_notch, notch_states[i] = lfilter(b_notch, a_notch, x_band, zi=notch_states[i])
        # Add to buffer
        channel_buffers[i].append(x_notch[0])

    #     if len(channel_buffers[i]) == buffer_size:
    #         plot_lines[i].set_ydata(channel_buffers[i])
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    #
    # Classify every 0.5s
    if all(len(buf) == buffer_size for buf in channel_buffers):
        if time.time() - last_print_time > window_duration:
            window = pd.DataFrame(
                {f" EXG Channel {i}": list(channel_buffers[i]) for i in range(8)}
            )
            prediction = classify_real_time(window, model, scaler)
            print("Predicted class:", POSE_MAP.get(prediction, prediction))
            last_print_time = time.time()


def classify_real_time(window, model, scaler):
    features = extract_features(window)
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]


# ======= Start Streaming =======
if __name__ == "__main__":
    # For controlling print rate
    last_print_time = 0
    # * This is hardcoded to my specific MAC port as OpenBCIs find port function was not working for me, this will need to be changed for other systems
    board = OpenBCICyton(port="/dev/tty.usbserial-DM03H689")
    for ch in range(1, 9):
        cmd = f"x{ch}0400000X"
        board.write_command(cmd)
    print("SRB2 disabled on all 8 channels.")

    print("Starting stream...")
    board.start_stream(handle_stream)
