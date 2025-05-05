import time
from collections import deque

import numpy as np
import pandas as pd
from pyOpenBCI import OpenBCICyton
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count

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


def train_model(data, window_duration=1, overlap_percentage=0.125, sample_rate=250):
    window_size = int(window_duration * sample_rate)
    step = int(window_size * (1 - overlap_percentage))
    X, y = [], []

    for start in range(0, len(data) - window_size, step):
        window = data.iloc[start : start + window_size]
        features = extract_features(window)
        X.append(features)
        y.append(window["class"].mode()[0])

    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    print(f"Training Accuracy: {model.score(X_scaled, y)*100:.2f}%")

    return model, scaler


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
model, scaler = train_model(filtered_data)


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

    # Classify every 0.5s
    if all(len(buf) == buffer_size for buf in channel_buffers):
        if time.time() - last_print_time > 0.5:
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

    board = OpenBCICyton(port="/dev/tty.usbserial-DM03H689")
    print("Starting stream...")
    board.start_stream(handle_stream)
