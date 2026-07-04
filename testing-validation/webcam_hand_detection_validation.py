import cv2
import mediapipe as mp
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import matplotlib.pyplot as plt
import threading

# Add the project root directory to sys.path so we can import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.hand import hand_landmarks
from utils.pose import pose_landmarks
from utils.fpga.fpga_serial import FPGASerial
from utils.fpga.fpga_packet import FPGAPacket
from utils.opencv.webcam_window import WebcamWindow
from utils.opencv.skeleton_window import SkeletonWindow

def save_performance_graph(time_log, ema_log, output_path):
    """Generates and saves the performance graph in a background thread."""
    if not time_log or not ema_log:
        print("No performance data collected. Graph was not generated.")
        return

    print("Generating performance graph in a background thread...")
    # Set high-quality styling
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Plot the EMA values over time
    plt.plot(time_log, ema_log, label='Timp de execuție', color='#007acc', linewidth=2)
    
    # Draw a line at 33 ms
    plt.axhline(y=33.0, color='#e53935', linestyle='--', linewidth=1.5, label='Timpul ideal de execuție (33 ms)')
    
    # Decorate the graph
    plt.title('Măsurarea timpului de execuție', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Timp parcurs (s)', fontsize=12)
    plt.ylabel('Valoare timp execuție (ms)', fontsize=12)
    
    # Customize grid and ticks
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Add a legend
    plt.legend(loc='upper right', frameon=True, shadow=False)
    
    # Save the graph
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nGraph successfully saved to: {output_path}")
    plt.close()

SCALE = 1000  # Milliseconds

def main():
    # Initialize Serial Port (COM3)
    fpga = FPGASerial(port='COM3', baudrate=115200)
    # Initialize Packet Handler
    fpga_pkt = FPGAPacket()

    # Define absolute model paths based on the project root
    hand_model_path = os.path.join(project_root, "models", "hand_landmarker.task")
    pose_model_path = os.path.join(project_root, "models", "pose_landmarker_lite.task")

    # Initialize landmarkers in VIDEO mode
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
    window = WebcamWindow("Webcam Hand Detection (Validation)")
    visualizer_3d = SkeletonWindow("3D Hand Skeleton (Validation)")

    print("--- Starting Webcam Hand Detection (Validation Mode) ---")
    print("Press 'Esc' or close the window to quit.")
    print("Press WASD to rotate skeleton view.")

    # Time tracking lists for plotting
    time_log = []
    ema_log = []

    start_time = time.perf_counter()
    last_time = start_time

    # Moving average variables
    avg_ms = 33.3
    alpha = 0.1

    # Use ThreadPoolExecutor for parallel inference
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while cap.isOpened():
            current_time = time.perf_counter()
            raw_delta_ms = (current_time - last_time) * SCALE
            last_time = current_time

            # Update moving average
            avg_ms = (1 - alpha) * avg_ms + alpha * raw_delta_ms

            # Log elapsed time (in seconds) and the calculated EMA (in ms)
            elapsed_seconds = current_time - start_time
            time_log.append(elapsed_seconds)
            ema_log.append(avg_ms)

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
                    fpga.send_packet(packet)
                    # Receive Feedback
                    feedback_packet = fpga.receive_packet()

                    # If no real serial or no response, fall back to simulation for UI
                    if not feedback_packet:
                        sim_payload = packet[1:9]
                        # Decoder expects XOR checksum of payload bytes
                        sim_checksum = 0
                        for b in sim_payload:
                            sim_checksum ^= b
                        feedback_packet = bytes([FPGAPacket.HEADER_RX]) + sim_payload + bytes([sim_checksum])

                    decoded = fpga_pkt.decode_fpga_packet(feedback_packet)

            # Update UI
            skeleton_canvas = visualizer_3d.render_from_packet(decoded)
            visualizer_3d.show(skeleton_canvas)
            window.draw_info(frame, decoded, avg_ms)
            window.show(frame)

            # Poll keys
            key = window.poll_key()
            if key in (27, ord('q')):
                break
            elif key == ord('a'):
                visualizer_3d.angle_y -= 10
            elif key == ord('d'):
                visualizer_3d.angle_y += 10
            elif key == ord('w'):
                visualizer_3d.angle_x -= 10
            elif key == ord('s'):
                visualizer_3d.angle_x += 10
            elif key == ord('z'):
                visualizer_3d.angle_z -= 10
            elif key == ord('x'):
                visualizer_3d.angle_z += 10

    except KeyboardInterrupt:
        print("\nInterrupted by user. Generating graph and exiting...")
    finally:
        # Cleanup
        executor.shutdown()
        fpga.close()
        landmarker.close()
        pose.close()
        cap.release()
        window.close()
        visualizer_3d.close()

        # Generate and save the graph asynchronously on a different thread
        if time_log and ema_log:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(output_dir, 'latency_ema_graph.png')
            
            # Start the graph saving thread (pass copies of logs to prevent race conditions)
            graph_thread = threading.Thread(
                target=save_performance_graph,
                args=(time_log.copy(), ema_log.copy(), output_path),
                daemon=False
            )
            graph_thread.start()
        else:
            print("No performance data collected. Graph was not generated.")

if __name__ == "__main__":
    main()
