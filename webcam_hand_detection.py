import cv2
import mediapipe as mp
import time
import struct
from concurrent.futures import ThreadPoolExecutor

from utils.hand import hand_landmarks
from utils.pose import pose_landmarks
from utils.fpga.fpga_serial import FPGASerial
from utils.fpga.fpga_packet import FPGAPacket
from utils.opencv.webcam_window import WebcamWindow
from utils.opencv.skeleton_window import SkeletonWindow

SCALE = 1000 # Milliseconds

def main():
    # Initialize Serial Port (COM3)
    fpga = FPGASerial(port='COM16', baudrate=115200)
    # Initialize Packet Handler
    fpga_pkt = FPGAPacket()

    # Initialize landmarkers in VIDEO mode
    landmarker = hand_landmarks.create_hand_landmarker(
        running_mode=hand_landmarks.VisionRunningMode.VIDEO
    )
    pose = pose_landmarks.create_pose_landmarker()

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
    window = WebcamWindow("Webcam Hand Detection")
    visualizer_3d = SkeletonWindow("3D Hand Skeleton")

    print("--- Starting Webcam Hand Detection ---")
    print("Press 'Esc' or close the window to quit.")
    print("Press WASD to rotate skeleton view.")

    last_time = time.perf_counter()

    # Moving average variables
    avg_ms = 33.3
    alpha = 0.1

    # Use ThreadPoolExecutor for parallel inference
    executor = ThreadPoolExecutor(max_workers=2)

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
            elbow_angle = 0 # Default straight
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
                # Ask the FPGA for feedback: it only replies to a 0xFE frame.
                # Echo the same payload so it can't zero the servo command.
                fpga.send_packet(fpga_pkt.create_read_request(packet))
                # Read the real AD7124 measurements it sends back.
                feedback_packet = fpga.receive_packet()

                if feedback_packet:
                    decoded = fpga_pkt.decode_fpga_packet(feedback_packet)
                else:
                    decoded = fpga_pkt.decode_fpga_packet_simulation(packet)

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

    # Cleanup
    executor.shutdown()
    fpga.close()
    landmarker.close()
    pose.close()
    cap.release()
    window.close()
    visualizer_3d.close()

if __name__ == "__main__":
    main()
