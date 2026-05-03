from enum import IntEnum

class HandJoint(IntEnum):
    """
    Data model for hand joint indices and connections.
    Based on the MediaPipe Hand Landmark model.
    """
    # Wrist
    WRIST = 0
    # Thumb
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    # Index
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    # Middle
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    # Ring
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    # Pinky
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

# Official MediaPipe Connections
HAND_CONNECTIONS = [
    # Thumb
    (HandJoint.WRIST, HandJoint.THUMB_CMC), (HandJoint.THUMB_CMC, HandJoint.THUMB_MCP),
    (HandJoint.THUMB_MCP, HandJoint.THUMB_IP), (HandJoint.THUMB_IP, HandJoint.THUMB_TIP),
    # Index
    (HandJoint.WRIST, HandJoint.INDEX_MCP), (HandJoint.INDEX_MCP, HandJoint.INDEX_PIP),
    (HandJoint.INDEX_PIP, HandJoint.INDEX_DIP), (HandJoint.INDEX_DIP, HandJoint.INDEX_TIP),
    # Middle
    (HandJoint.WRIST, HandJoint.MIDDLE_MCP), (HandJoint.MIDDLE_MCP, HandJoint.MIDDLE_PIP),
    (HandJoint.MIDDLE_PIP, HandJoint.MIDDLE_DIP), (HandJoint.MIDDLE_DIP, HandJoint.MIDDLE_TIP),
    # Ring
    (HandJoint.WRIST, HandJoint.RING_MCP), (HandJoint.RING_MCP, HandJoint.RING_PIP),
    (HandJoint.RING_PIP, HandJoint.RING_DIP), (HandJoint.RING_DIP, HandJoint.RING_TIP),
    # Pinky
    (HandJoint.WRIST, HandJoint.PINKY_MCP), (HandJoint.PINKY_MCP, HandJoint.PINKY_PIP),
    (HandJoint.PINKY_PIP, HandJoint.PINKY_DIP), (HandJoint.PINKY_DIP, HandJoint.PINKY_TIP)
]