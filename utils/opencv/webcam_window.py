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
            fingers_text = f"P:{decoded['pinky']} R:{decoded['ring']} M:{decoded['middle']} I:{decoded['index']} T:{decoded['thumb']}"
            cv2.putText(frame, fingers_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            servos_text = f"Palm:{decoded['thumb_palm']} Tor:{decoded['thumb_torsion']} Wri:{decoded['wrist']} Elb:{decoded['elbow']}"
            cv2.putText(frame, servos_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"AVG: {avg_ms:.1f}ms", (frame.shape[1] - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
