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

### 1. Create a Virtual Environment

It is highly recommended to run this project in a Python virtual environment.

```bash
# Navigate to the project directory
cd /path/to/your/project

# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all the required Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```
