import numpy as np
from .hand_model import HandJoint

SCALE = 1000  # Conversion from meters to millimeters

# Finger joint mapping for flexion calculation
# Format: (Base Joint, Middle Joint/Vertex, Tip-ward Joint)
FINGER_JOINT_MAP = {
    0: (HandJoint.THUMB_MCP,  HandJoint.THUMB_IP,  HandJoint.THUMB_TIP),
    1: (HandJoint.INDEX_MCP,  HandJoint.INDEX_PIP,  HandJoint.INDEX_DIP),
    2: (HandJoint.MIDDLE_MCP, HandJoint.MIDDLE_PIP, HandJoint.MIDDLE_DIP),
    3: (HandJoint.RING_MCP,   HandJoint.RING_PIP,   HandJoint.RING_DIP),
    4: (HandJoint.PINKY_MCP,  HandJoint.PINKY_PIP,  HandJoint.PINKY_DIP)
}

def calculate_angle(p1, p2, p3):
    """
    Calculates the raw 3D angle at vertex p2 formed by vectors (p2->p1) and (p2->p3).

    Args:
        p1, p2, p3: MediaPipe landmarks with x, y, z attributes.

    Returns:
        float: Angle in degrees in the range [0, 180].
    """
    # Create vectors relative to the vertex p2
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    # Calculate vector magnitudes
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle edge case: zero-length vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    # Compute angle using dot product and arccos
    dot_product = np.dot(v1 / norm_v1, v2 / norm_v2)
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

def get_distance(p1, p2):
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        p1, p2: MediaPipe landmarks with x, y, z attributes.

    Returns:
        float: The distance between the points.
    """
    return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]))

def get_finger_flexion(world_landmarks, finger_type: int):
    """
    Calculates the flexion of a specific finger.

    Args:
        world_landmarks (list): List of world landmarks from MediaPipe result.
        finger_type (int): Finger index (0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky).

    Returns:
        float: Flexion angle (0 = Straight, 180 = Fully bent).
    """
    if not world_landmarks:
        return -1
    
    indices = FINGER_JOINT_MAP[finger_type]
    raw_angle = calculate_angle(
        world_landmarks[indices[0]],
        world_landmarks[indices[1]],
        world_landmarks[indices[2]]
    )

    # Fingers are straight at 180 deg raw, so flexion is 180 - raw
    return 180 - raw_angle

def calculate_finger_spread(world_landmarks):
    """
    Measures the distance between the Index Tip and Pinky Tip.

    Args:
        world_landmarks (list): List of world landmarks.

    Returns:
        int: Scaled distance in arbitrary units (approx. mm).
    """
    if not world_landmarks:
        return -1
    
    dist_m = get_distance(world_landmarks[HandJoint.INDEX_TIP],
                          world_landmarks[HandJoint.PINKY_TIP])

    return int(dist_m * SCALE)

def calculate_thumb_opposition(world_landmarks):
    """
    Measures the distance between the Thumb Tip and Pinky Base.

    Args:
        world_landmarks (list): List of world landmarks.

    Returns:
        int: Scaled distance in arbitrary units (approx. mm).
    """
    if not world_landmarks:
        return -1

    dist_m = get_distance(world_landmarks[HandJoint.THUMB_TIP],
                          world_landmarks[HandJoint.PINKY_MCP])

    return int(dist_m * SCALE)

def calculate_wrist_flexion(world_landmarks):
    """
    Approximates wrist flexion using the angle between the vertical and palm vector.

    Args:
        world_landmarks (list): List of world landmarks.

    Returns:
        int: Angle representing wrist flexion/extension (approx. 90 is neutral).
    """
    if not world_landmarks:
        return 0

    wrist = world_landmarks[HandJoint.WRIST]
    mcp = world_landmarks[HandJoint.MIDDLE_MCP]

    # Vector from Wrist to Middle Finger Base
    v = np.array([mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z])
    
    # Calculate angle in Y-Z plane (nodding motion)
    # MediaPipe Y increases downwards, so we negate it for standard verticality
    angle_rad = np.arctan2(v[2], -v[1])
    return int(90 + np.degrees(angle_rad))
