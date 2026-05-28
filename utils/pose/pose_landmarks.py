import cv2
import mediapipe as mp
import numpy as np
from utils.hand.hand_physics import calculate_angle

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmark = mp.tasks.vision.PoseLandmark

MODEL_PATH = "models/pose_landmarker_lite.task"

# Arm landmark indices per side
_ARM_LANDMARKS = {
    "Left":  (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    "Right": (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
}

def create_pose_landmarker(model_path=MODEL_PATH, running_mode=VisionRunningMode.VIDEO):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
    )
    return PoseLandmarker.create_from_options(options)

def get_elbow_angle(pose_landmarks, handedness="Right"):
    """
    Calculates the elbow flexion angle using 2D projection for stability.
    Returns the raw interior angle (180=straight, 0=fully bent).
    """
    if not pose_landmarks:
        return 180

    # Handle case-insensitive handedness from MediaPipe
    side = "Left" if handedness.capitalize() == "Left" else "Right"
    shoulder_idx, elbow_idx, wrist_idx = _ARM_LANDMARKS[side]

    shoulder = pose_landmarks[shoulder_idx]
    elbow = pose_landmarks[elbow_idx]
    wrist = pose_landmarks[wrist_idx]

    # Use 2D coordinates (x, y) for a more stable "visual" angle
    v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
    v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 180

    dot_product = np.dot(v1 / norm_v1, v2 / norm_v2)
    raw_angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    return int(np.clip(raw_angle, 0, 180))

def draw_arm_landmarks(image, pose_landmarks, handedness="Right"):
    """Draws shoulder-elbow-wrist connections on the image."""
    if not pose_landmarks:
        return image

    h, w = image.shape[:2]
    side = "Left" if handedness.capitalize() == "Left" else "Right"
    shoulder_idx, elbow_idx, wrist_idx = _ARM_LANDMARKS[side]

    points = []
    for idx in (shoulder_idx, elbow_idx, wrist_idx):
        lm = pose_landmarks[idx]
        points.append((int(lm.x * w), int(lm.y * h)))

    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
    for pt in points:
        cv2.circle(image, pt, 4, (0, 200, 255), -1)

    return image
