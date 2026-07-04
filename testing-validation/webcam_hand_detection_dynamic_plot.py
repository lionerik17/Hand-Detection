import cv2
import mediapipe as mp
import time
import random
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import threading
import matplotlib
# Force TkAgg backend for consistent behavior in background threads
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Add the project root directory to sys.path so we can import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.hand import hand_landmarks
from utils.pose import pose_landmarks
from utils.fpga.fpga_serial import FPGASerial
from utils.fpga.fpga_packet import FPGAPacket
from utils.opencv.webcam_window import WebcamWindow

SCALE = 1000  # Milliseconds

# Global tracking variable for current byte index to plot (0 to 7)
current_byte_index = 0

# A beautiful gradient color palette (8 distinct colors from Cool to Warm)
GRADIENT_COLORS = [
    "#3a86c8",  # Byte 1: Light Blue
    "#009cd3",  # Byte 2: Cyan
    "#00b2bb",  # Byte 3: Turquoise
    "#00c391",  # Byte 4: Emerald Green
    "#7ad05b",  # Byte 5: Lime Green
    "#ffd32a",  # Byte 6: Warm Yellow
    "#ff9f43",  # Byte 7: Soft Orange
    "#ff4757"   # Byte 8: Soft Red
]

# Shared metric names
BYTE_NAMES = ["Pinky", "Ring", "Middle", "Index", "Thumb Oppositon", "Thumb", "Wrist", "Elbow"]

def plot_thread_fn(stop_event, data_lock, frame_numbers, sent_history, recv_history):
    """Background thread that manages and updates the dynamic Matplotlib plot."""
    global current_byte_index
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.manager.set_window_title("Cadre trimise vs Cadre recepționate")
    
    # Initialize sent and received lines (thick received line, thin sent line)
    # Using a neutral dark slate color for Sent to contrast beautifully with active gradient colors for Received
    line_sent, = ax.plot([], [], label="Trimis", color="#2c3e50", linewidth=2.0, linestyle="--", marker="o", markersize=4.5, alpha=0.7)
    line_recv, = ax.plot([], [], label="Recepționat", color=GRADIENT_COLORS[0], linewidth=3.0, marker="o", markersize=4.5)
    
    ax.set_ylim(-10, 190)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    
    # Keypress callback for Matplotlib window
    def on_key(event):
        global current_byte_index
        if event.key in ['1', '2', '3', '4', '5', '6', '7', '8']:
            with data_lock:
                current_byte_index = int(event.key) - 1
            print(f"[Plot Window] Switched view to: {BYTE_NAMES[current_byte_index]} (Byte {current_byte_index + 1})")
            
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    while not stop_event.is_set():
        with data_lock:
            if not frame_numbers:
                time.sleep(0.05)
                continue
            
            # Copy last 100 frames to keep plot clean and fast
            x = list(frame_numbers[-100:])
            y_sent = list(sent_history[current_byte_index][-100:])
            y_recv = list(recv_history[current_byte_index][-100:])
            curr_byte = current_byte_index
            
        ax.set_title(f"Byte curent: {BYTE_NAMES[curr_byte]} (Byte {curr_byte+1})\nApăsați tastele 1-8 pentru a schimba byte-ul", fontsize=12, fontweight='bold')
        ax.set_xlabel("Număr cadru")
        ax.set_ylabel("Valoare decodificată din cadru")
        
        # Update lines data
        line_sent.set_data(x, y_sent)
        line_recv.set_data(x, y_recv)
        
        # Update colors based on the current byte gradient
        line_color = GRADIENT_COLORS[curr_byte]
        line_recv.set_color(line_color)
        
        # Adjust x-axis dynamically
        ax.set_xlim(x[0], x[-1] + 1)
        
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            # Handle if the window is closed manually
            break
            
        time.sleep(0.03)  # Loop at roughly 30 FPS
        
    plt.close(fig)

def main():
    global current_byte_index

    # Initialize Serial Port (COM3)
    fpga = FPGASerial(port='COM3', baudrate=115200)
    # Initialize Packet Handler
    fpga_pkt = FPGAPacket()

    # Initialize landmarkers in VIDEO mode
    hand_model_path = os.path.join(project_root, "models", "hand_landmarker.task")
    pose_model_path = os.path.join(project_root, "models", "pose_landmarker_lite.task")

    landmarker = hand_landmarks.create_hand_landmarker(
        running_mode=hand_landmarks.VisionRunningMode.VIDEO,
        model_path=hand_model_path
    )
    pose = pose_landmarks.create_pose_landmarker(
        model_path=pose_model_path
    )

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        fpga.close()
        return

    # Set resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize Windows
    window = WebcamWindow("Webcam Hand Detection (Plot Mode)")

    print("--- Starting Webcam Hand Detection with Dynamic Plot ---")
    print("Press Esc/'q' to quit.")
    print("Press 1-8 to toggle showing corresponding byte sent/received.")

    last_time = time.perf_counter()

    # Moving average variables
    avg_ms = 33.3
    alpha = 0.1

    # Threading setup for Matplotlib dynamic plotting
    stop_event = threading.Event()
    data_lock = threading.Lock()
    frame_numbers = []
    sent_history = [[] for _ in range(8)]
    recv_history = [[] for _ in range(8)]

    # Initial telemetry values
    curr_sent = [0] * 8
    curr_recv = [0] * 8
    frame_count = 0

    # Start the plotting thread
    plot_thread = threading.Thread(
        target=plot_thread_fn,
        args=(stop_event, data_lock, frame_numbers, sent_history, recv_history)
    )
    plot_thread.start()

    # Use ThreadPoolExecutor for parallel inference
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while cap.isOpened():
            current_time = time.perf_counter()
            raw_delta_ms = (current_time - last_time) * SCALE
            last_time = current_time

            # Update moving average
            avg_ms = (1 - alpha) * avg_ms + alpha * raw_delta_ms

            success, frame = cap.read()
            if not success:
                continue

            # Optimize preprocessing: flip and convert in one go
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Get the current timestamp
            frame_timestamp_ms = int(time.time() * SCALE)

            # Parallel inference: Hand and Pose detection
            future_hand = executor.submit(landmarker.detect_for_video, mp_image, frame_timestamp_ms)
            future_pose = executor.submit(pose.detect_for_video, mp_image, frame_timestamp_ms)

            result = future_hand.result()
            pose_result = future_pose.result()

            decoded = None
            if result.hand_landmarks:
                # Draw hand landmarks
                hand_landmarks.draw_landmarks(frame, result)
                primary_hand_world = result.hand_world_landmarks[0]

                # Get elbow angle from pose
                elbow_angle = 0  # Default straight
                if pose_result.pose_landmarks:
                    handedness = result.handedness[0][0].category_name
                    elbow_angle = pose_landmarks.get_elbow_angle(
                        pose_result.pose_landmarks[0], handedness
                    )
                    pose_landmarks.draw_arm_landmarks(
                        frame, pose_result.pose_landmarks[0], handedness
                    )

                # Create and send FPGA packet
                packet = fpga_pkt.create_fpga_packet(primary_hand_world, elbow_angle)

                if packet:
                    # Update local state of data sent
                    curr_sent = list(packet[1:9])

                    fpga.send_packet(packet)
                    # Receive Feedback
                    feedback_packet = fpga.receive_packet()

                    # If no real serial or no response, fall back to simulation for UI
                    if not feedback_packet:
                        sim_payload = packet[1:9]
                        # Introduce a small fluctuation to simulated received values only if not connected to COM3
                        if not fpga.is_connected():
                            sim_payload = bytearray(sim_payload)
                            for i in range(len(sim_payload)):
                                fluctuation = random.randint(-5, 5)
                                sim_payload[i] = max(0, min(180, sim_payload[i] + fluctuation))
                            sim_payload = bytes(sim_payload)
                        # Decoder expects XOR checksum of payload bytes
                        sim_checksum = 0
                        for b in sim_payload:
                            sim_checksum ^= b
                        feedback_packet = bytes([FPGAPacket.HEADER_RX]) + sim_payload + bytes([sim_checksum])

                    decoded = fpga_pkt.decode_fpga_packet(feedback_packet)
                    if decoded:
                        # Update local state of data received
                        curr_recv = [
                            decoded["pinky"],
                            decoded["ring"],
                            decoded["middle"],
                            decoded["index"],
                            decoded["thumb_palm"],
                            decoded["thumb"],
                            decoded["wrist"],
                            decoded["elbow"]
                        ]

            # Append the current frame's sent/received telemetry to data history
            with data_lock:
                frame_count += 1
                frame_numbers.append(frame_count)
                for j in range(8):
                    sent_history[j].append(curr_sent[j])
                    recv_history[j].append(curr_recv[j])

            # Update UI
            window.draw_info(frame, decoded, avg_ms)
            window.show(frame)

            # Poll keys
            key = window.poll_key()
            if key in (27, ord('q')):
                break
            elif ord('1') <= key <= ord('8'):
                with data_lock:
                    current_byte_index = key - ord('1')
                print(f"[OpenCV Window] Switched view to: {BYTE_NAMES[current_byte_index]} (Byte {current_byte_index + 1})")


    finally:
        # Cleanup main threads and stop background plot thread
        print("\nCleaning up and closing windows...")
        stop_event.set()
        plot_thread.join(timeout=2.0)
        executor.shutdown()
        fpga.close()
        landmarker.close()
        pose.close()
        cap.release()
        window.close()

if __name__ == "__main__":
    main()
