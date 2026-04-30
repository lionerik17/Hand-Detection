import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MODEL_PATH = "models/hand_landmarker.task"

def create_hand_landmarker(model_path: str = MODEL_PATH, num_hands: int = 1, running_mode=VisionRunningMode.IMAGE):
    """
    Initializes and returns a MediaPipe Hand Landmarker object.

    Args:
        model_path (str): The file path to the hand landmarker task model.
        num_hands (int): Maximum number of hands to detect.
        running_mode (VisionRunningMode): The operating mode (IMAGE, VIDEO, or LIVE_STREAM).

    Returns:
        HandLandmarker: A configured MediaPipe Hand Landmarker instance.
    """
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_hands=num_hands,
    )
    return HandLandmarker.create_from_options(options)

def draw_landmarks(image, detection_result):
    """
    Draws detected hand landmarks and connections on the provided image.

    Args:
        image (numpy.ndarray): The BGR image to draw on.
        detection_result (HandLandmarkerResult): The detection results from MediaPipe.

    Returns:
        numpy.ndarray: The image with landmarks drawn.
    """
    if not detection_result or not detection_result.hand_landmarks:
        return image

    # Iterate through all detected hands
    for hand_landmarks in detection_result.hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return image
