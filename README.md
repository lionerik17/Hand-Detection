# Hand Detection & FPGA Communication

This project implements a real-time hand detection and gesture analysis system using **MediaPipe** and **OpenCV**. It extracts hand landmarks, calculates finger flexion and specialized metrics (spread, opposition, wrist flexion), and transmits this data to an external FPGA device via Serial communication.

## Features

- **Real-time Detection**: Uses MediaPipe Landmarker for high-performance hand tracking via webcam.
- **Bi-directional Serial Communication**: Sends 10-byte data packets to an FPGA and receives feedback.
- **Gesture Analysis**: Calculates angles for all five fingers and specialized metrics (Thumb Opposition, Finger Spread, Wrist Flexion).
- **Modular Architecture**: Organized for better maintainability with dedicated modules for hand physics, serial communication, and UI management.

## Project Structure

```text
hand-detection/
├── models/                   # AI Model files
│   └── hand_landmarker.task  # MediaPipe Hand Landmarker task file
├── utils/
│   ├── fpga/                 # Serial & Packet protocol
│   │   ├── fpga_packet.py    # Binary packet construction and decoding
│   │   └── fpga_serial.py    # Serial communication class (initializes COM)
│   ├── hand/                 # MediaPipe & Physics logic
│   │   ├── hand_landmarks.py # Hand landmark detection
│   │   ├── hand_model.py     # Joint mapping and skeleton connections
│   │   └── hand_physics.py   # Metrics (angles, distances, flexion)
│   └── opencv/               # UI & Window management
│       └── window.py         # Helper class for OpenCV windows
├── requirements.txt          # Python dependencies
└── webcam_hand_detection.py  # Main entry point for real-time detection
```

## Installation

1. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Serial Protocol (10-byte Packet)

The system communicates using a structured binary protocol at **115200 baud**.

| Byte | Field | Description |
| :--- | :--- | :--- |
| 0 | Header | `0xFF` (TX from PC) or `0xFE` (RX from FPGA) |
| 1 | Thumb | Flexion value (0-255) |
| 2 | Index | Flexion value (0-255) |
| 3 | Middle | Flexion value (0-255) |
| 4 | Ring | Flexion value (0-255) |
| 5 | Pinky | Flexion value (0-255) |
| 6 | Opposition | Thumb-Index opposition metric (0-255) |
| 7 | Spread | Finger spread metric (0-255) |
| 8 | Wrist | Wrist flexion/extension (0-255) |
| 9 | Checksum | 8-bit sum of bytes 1-8 modulo 256 |

## Usage

### Webcam Detection
To start the real-time detection and serial transmission:
```bash
python webcam_hand_detection.py
```
*   **Default Port**: `COM3` (can be modified in `webcam_hand_detection.py`).
*   **Controls**: 
    *   Press **`Esc`** or **`q`** to exit.
    *   Closing the window using the **`X`** button also terminates the program.
*   **Console Logging**: 
    *   `TX -> FPGA`: Data sent to the device.
    *   `RX <- FPGA`: Response received from the device.
    *   `[SIM]`: Indicates simulation mode (used if no device is connected).
