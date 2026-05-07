import cv2
import mediapipe as mp
import time

from utils.hand import hand_landmarks
from utils.fpga import fpga_packet
from utils.fpga.fpga_serial import FPGASerial
from utils.opencv.webcam_window import WebcamWindow
from utils.opencv.skeleton_window import SkeletonWindow

SCALE = 1000 # Milliseconds

def main():
    # Initialize Serial Port (COM3)
    fpga = FPGASerial(port='COM3', baudrate=115200)

    # Initialize the landmarker in VIDEO mode
    landmarker = hand_landmarks.create_hand_landmarker(
        running_mode=hand_landmarks.VisionRunningMode.VIDEO
    )

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        fpga.close()
        return
    
    # Initialize Windows
    window = WebcamWindow("Webcam Hand Detection")
    visualizer_3d = SkeletonWindow("3D Hand Skeleton")

    print("--- Starting Webcam Hand Detection ---")
    print("Press 'Esc' or close the window to quit.")

    last_time = time.perf_counter()

    # Moving average variables
    avg_ms = 33.3
    alpha = 0.1

    while cap.isOpened():
        current_time = time.perf_counter()
        raw_delta_ms = (current_time - last_time) * SCALE
        last_time = current_time

        # Update moving average
        avg_ms = (1 - alpha) * avg_ms + alpha * raw_delta_ms

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame horizontally for a more natural selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Get the current timestamp
        frame_timestamp_ms = int(time.time() * SCALE)

        # Detect landmarks in the frame
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        decoded = None
        if result.hand_landmarks:
            # Draw landmarks on the frame
            hand_landmarks.draw_landmarks(frame, result)

            # Extract world landmarks for the detected hand
            primary_hand_world = result.hand_world_landmarks[0]
            
            # 1. Create the FPGA packet (Header 0xFF)
            packet = fpga_packet.create_fpga_packet(primary_hand_world)

            if packet:
                # 1. Visualize sent data in console
                status_prefix = "TX -> FPGA" if fpga.is_connected() else "[SIM] TX -> FPGA"
                print(f"{status_prefix}: {packet.hex().upper()}")

                # 2. Send to Serial Port
                fpga.send_packet(packet)

                # 3. Receive Feedback (Real or Simulated)
                feedback_packet = fpga.receive_packet()
                
                # If no real serial or no response, fall back to simulation for UI visualization
                if not feedback_packet:
                    feedback_packet = b'\xFE' + packet[1:]
                
                decoded = fpga_packet.decode_fpga_packet(feedback_packet)
                
                if decoded:
                    if fpga.is_connected() and feedback_packet[0] == 0xFE:
                         print(f"RX <- FPGA: {feedback_packet.hex().upper()}")

        # Update 3D Visualization using DECODED packet data
        skeleton_canvas = visualizer_3d.render_from_packet(decoded)
        visualizer_3d.show(skeleton_canvas)

        # Show detection status and metrics on screen
        window.draw_info(frame, decoded, avg_ms)
        
        # Display the frame
        window.show(frame)

        if window.should_close() or visualizer_3d.should_close():
            break
    
    # Cleanup
    fpga.close()
    landmarker.close()
    cap.release()
    window.close()
    visualizer_3d.close()

if __name__ == "__main__":
    main()
