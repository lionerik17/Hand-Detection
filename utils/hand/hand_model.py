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

    @classmethod
    def get_finger_bases(cls):
        """Returns the base (MCP/CMC) joints for all five fingers."""
        return [cls.THUMB_CMC, cls.INDEX_MCP, cls.MIDDLE_MCP, cls.RING_MCP, cls.PINKY_MCP]

    @classmethod
    def get_finger_chains(cls):
        """Returns the full 4-joint chains for each finger."""
        return [
            (cls.THUMB_CMC, cls.THUMB_MCP, cls.THUMB_IP, cls.THUMB_TIP),
            (cls.INDEX_MCP, cls.INDEX_PIP, cls.INDEX_DIP, cls.INDEX_TIP),
            (cls.MIDDLE_MCP, cls.MIDDLE_PIP, cls.MIDDLE_DIP, cls.MIDDLE_TIP),
            (cls.RING_MCP, cls.RING_PIP, cls.RING_DIP, cls.RING_TIP),
            (cls.PINKY_MCP, cls.PINKY_PIP, cls.PINKY_DIP, cls.PINKY_TIP)
        ]

    @classmethod
    def get_palm_indices(cls):
        """Returns the indices forming the palm polygon."""
        return [cls.WRIST, cls.THUMB_CMC, cls.INDEX_MCP, cls.MIDDLE_MCP, cls.RING_MCP, cls.PINKY_MCP]

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