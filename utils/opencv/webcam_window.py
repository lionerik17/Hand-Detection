import cv2
from .base_window import OpenCVWindow

class WebcamWindow(OpenCVWindow):
    """
    Window specialized for displaying webcam feed with overlay info.
    """
    def __init__(self, window_name="Webcam Hand Detection"):
        super().__init__(window_name)

    def draw_info(self, frame, decoded, avg_ms):
        """Draws telemetry and status on the webcam frame."""
        if decoded:
            cv2.putText(frame, "Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Line 1: Finger Flexions
            fingers_text = f"Fingers: T:{decoded['thumb']} I:{decoded['index']} M:{decoded['middle']} R:{decoded['ring']} P:{decoded['pinky']}"
            cv2.putText(frame, fingers_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Line 2: Specialized Metrics
            metrics_text = f"Metrics: Opp:{decoded['opposition']} Spr:{decoded['spread']} Wri:{decoded['wrist']}"
            cv2.putText(frame, metrics_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Average time to compute landmarks and to create, send and receive packets
            cv2.putText(frame, f"AVG: {avg_ms:.1f}ms", (frame.shape[1] - 180, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
