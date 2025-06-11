# Real-Time sEMG Gesture Classification Application

This application provides a graphical user interface to capture, process, and classify surface electromyography (sEMG) signals from an OpenBCI Cyton board in real-time. It uses a pre-trained machine learning model to predict one of six hand and wrist gestures.

## Features

* **Live Data Visualisation:** Plots all 8 sEMG channels in real-time.
* **Real-Time Classification:** Predicts gestures (Rest, Fist, Flexion, Extension, Radial Deviation, Ulnar Deviation) with low latency.
* **Stable Predictions:** Implements a majority-voting buffer to smooth the output and reduce classification jitter.
* **Configurable:** Allows the user to select from different pre-trained models and specify the arm being used (Left/Right) for correct channel mapping.

## Hardware Requirements

* OpenBCI Cyton Biosensing Board
* RFDuino USB Dongle
* A custom forearm armband with 8 bipolar sEMG electrodes connected to all 8 channels of the Cyton board.
* A reference electrode connected to the BIAS pin.

## Setup Instructions

### 1. Clone repository and Open the directory in your terminal

```
git clone https://github.com/LilOz/emg-realtime-classification
```

```
cd emg-realtime-classification
```

### 2. Create a virtual environment:

```
python -m venv venv
```

or

```
python3 -m venv venv
```

### 3. Activate virtual environment:

Windows

```
venv\Scripts\activate
```

MacOS / Linux

```
source venv/bin/activate
```

After doing this you should see (venv) in your terminal

### 4. Install Dependencies:

```
pip install -r requirements.txt
```

## Running the Application

### Launch the Script

With your virtual environment activated, run the script from your terminal:

`python realtime_emg.py`

### Configure Session

The script will prompt you in the terminal for two configuration steps:

1.  **Select a model:** It will list all available models from the `models` directory. Enter the number corresponding to the model you wish to use.
2.  **Select the arm:** Enter `L` for the left arm or `R` for the right arm.

Once configured, the GUI will launch, connect to the board, and begin real-time classification.
