import threading
import time
from collections import deque, Counter
from itertools import combinations
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from pyOpenBCI import OpenBCICyton
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
from tensorflow.keras.models import load_model
from feature_extractors import ln_rms, aac, mavs, ssc, wamp, skewness, ssi
import matplotlib.image as mpimg

PRINT_INTERVAL = 0.512
PREDICTION_BUFFER_SIZE = 8
CLASSIFY_INTERVAL = PRINT_INTERVAL / PREDICTION_BUFFER_SIZE
SCALE_FACTOR_EEG = (4500000) / 24 / (2**23 - 1)  # uV/count
POSE_IMAGES = {
    1: mpimg.imread("imgs/rest.png"),
    2: mpimg.imread("imgs/fist.png"),
    3: mpimg.imread("imgs/flexion.png"),
    4: mpimg.imread("imgs/extension.png"),
    5: mpimg.imread("imgs/radial.png"),
    6: mpimg.imread("imgs/ulnar.png"),
}
POSE_MAP = {
    1: "rest",
    2: "fist",
    3: "wrist flexion (down)",
    4: "wrist extension (up)",
    5: "radial deviation (right)",
    6: "ulnar deviation (left)",
}

# Ask user for arm selection before starting the stream
def select_arm():
    while True:
        choice = input("Select arm [L/R]: ").strip().lower()
        if choice in ["l", "r"]:
            return choice == "r"
        print("Invalid input. Please enter 'L' for left or 'R' for right.")

# ====== Model Selection ======
def select_model(models_dir="models", max_models=10):
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    subdirs = sorted(subdirs)[:max_models]

    if not subdirs:
        raise ValueError("No models found in the models directory.")

    print("Available Models:")
    for i, name in enumerate(subdirs):
        print(f"{i + 1}. {name}")

    idx = int(input(f"Select a model [1-{len(subdirs)}]: ")) - 1
    if not (0 <= idx < len(subdirs)):
        raise ValueError("Invalid selection.")

    selected_path = os.path.join(models_dir, subdirs[idx])
    model = load_model(os.path.join(selected_path, "model.h5"))
    scaler = joblib.load(os.path.join(selected_path, "scaler.pkl"))
    print(f"Loaded model: {subdirs[idx]}")
    return model, scaler

def extract_features(window, feature_extractors=[ln_rms, aac, mavs, ssc, wamp, skewness, ssi]):
    features = []
    channel_data = []

    for i in range(8):
        x = window[f' EXG Channel {i}'].values
        channel_data.append(x)
        for extractor in feature_extractors:
            result = extractor(x)
            features.extend(result if isinstance(result, (list, np.ndarray)) else [result])

    channel_data = np.array(channel_data)
    for i, j in combinations(range(8), 2):
        features.append(np.corrcoef(channel_data[i], channel_data[j])[0, 1])

    return features


# ====== Classification ======
def classify_real_time(window):
    features = extract_features(window)
    features_scaled = scaler.transform([features])
    probs = model.predict(features_scaled, verbose=0)
    predicted_class = np.argmax(probs, axis=1)[0]  # using softmax so choose the highest probability
    return predicted_class


# ====== Streaming Thread ======
def stream_thread(reverse_channels=False):
    global band_states, notch_states, last_print_time, last_classify_time

    def handle_stream(sample):
        global band_states, notch_states, last_print_time, last_classify_time

        raw_data = []
        for i in range(8):
            x = sample.channels_data[i] * SCALE_FACTOR_EEG
            x_band, band_states[i] = lfilter(b_band, a_band, [x], zi=band_states[i])
            x_notch, notch_states[i] = lfilter(b_notch, a_notch, x_band, zi=notch_states[i])
            x_notch[0] = np.clip(x_notch[0], -600, 600)
            raw_data.append(x_notch[0])

        if reverse_channels:
            raw_data = raw_data[::-1]  # Reverse channel order if on right arm

        for i, val in enumerate(raw_data):
            channel_buffers[i].append(val)
            plot_buffers[i].append(val)

        if all(len(buf) == buffer_size for buf in channel_buffers):
            if time.time() - last_classify_time > CLASSIFY_INTERVAL:
                window = pd.DataFrame(
                    {f" EXG Channel {i}": list(channel_buffers[i]) for i in range(8)}
                )
                prediction_buffer.append(classify_real_time(window))
                last_classify_time = time.time()
            if time.time() - last_print_time > PRINT_INTERVAL:
                prediction = Counter(prediction_buffer).most_common(1)[0][0]                
                print("Predicted class:", POSE_MAP.get(prediction + 1, prediction))
                last_print_time = time.time()

    last_print_time = 0
    last_classify_time = 0
    board = OpenBCICyton(port="/dev/tty.usbserial-DM03H689")
    for ch in range(1, 9):
        cmd = f"x{ch}0400000X"
        board.write_command(cmd)
    print("SRB2 disabled on all 8 channels.")

    print("Starting stream...")
    board.start_stream(handle_stream)


# ====== Parameters ======
fs = 250
window_duration = 0.128 
buffer_size = int(fs * window_duration)
time_step = 1 / fs
plot_buffer_len = 3 * fs  # Seconds of data per channel

lowcut, highcut, notch_freq, order = 20, 124, 50, 8
nyq = 0.5 * fs
b_band, a_band = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
b_notch, a_notch = iirnotch(notch_freq / nyq, 30)

# ====== Shared Buffers ======
channel_buffers = [deque(maxlen=buffer_size) for _ in range(8)]  # for classification
plot_buffers = [deque(maxlen=plot_buffer_len) for _ in range(8)] # for plotting

prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)

band_states = [lfilter_zi(b_band, a_band) * 0 for _ in range(8)]
notch_states = [lfilter_zi(b_notch, a_notch) * 0 for _ in range(8)]

# Load the selected model and scaler
model, scaler = select_model()

reverse_channels = select_arm()

# ====== Start Thread ======
threading.Thread(target=stream_thread, daemon=True, args=(reverse_channels,)).start()

# ====== Plotting ======
fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 14), sharex=False)
emg_axes = axes[:-1]
image_ax = axes[-1]

lines = []

for i, ax in enumerate(emg_axes):
    line, = ax.plot([], [], label=f"Channel {i+1}")
    lines.append(line)
    ax.set_ylim(-300, 300)  # Adjust based on expected signal range
    ax.set_xlim(0, plot_buffer_len)
    ax.set_ylabel(f"Ch {i+1}")
    ax.legend(loc="upper right")

emg_axes[-1].set_xlabel("Samples")

current_pose = 1  # default
img_display = image_ax.imshow(POSE_IMAGES[current_pose])
image_ax.axis('off')
# Add dynamic text label above the image
label_text = image_ax.text(
    0.5, 1.05, POSE_MAP[current_pose], ha='center', va='bottom',
    fontsize=14, fontweight='bold', transform=image_ax.transAxes
)

plt.tight_layout()

def animate(i):
    global current_pose
    for idx, line in enumerate(lines):
        y = list(plot_buffers[idx])
        x = list(range(len(y)))
        line.set_data(x, y)

    # Update prediction and corresponding image + label
    try:
        current_pose = Counter(prediction_buffer).most_common(1)[0][0] + 1
    except Exception:
        current_pose = 1
    img_display.set_data(POSE_IMAGES[current_pose])
    label_text.set_text(POSE_MAP[current_pose])

    return lines + [img_display, label_text]



ani = FuncAnimation(fig, animate, interval=50, blit=True)
plt.tight_layout()
plt.show()
