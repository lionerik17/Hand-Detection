import struct
from utils.hand import hand_physics

class FPGAPacket:
    """
    A class to handle the construction and decoding of binary data packets 
    for communication with an FPGA device over UART.
    Uses the struct module for robust binary framing.
    """

    HEADER_TX = 0xFF  # Write servo angles
    HEADER_RX = 0xFE  # Read-back from FPGA

    def __init__(self):
        # Format: 1 byte header, 8 bytes payload (servos), 1 byte checksum
        self.packet_format = ">B8BB"

    def calculate_checksum(self, payload):
        """Calculates XOR checksum of a list of bytes."""
        checksum = 0
        for val in payload:
            checksum ^= val
        return checksum

    def create_fpga_packet(self, hand_world_landmarks, elbow_angle: int) -> bytes:
        """
        Constructs a 10-byte packet for the FPGA.
        Format: [0xFF, Pinky, Ring, Middle, Index, ThumbPalm, Thumb, Wrist, Elbow, Checksum]
        """
        if not hand_world_landmarks:
            return None

        # Calculate flexion for all 5 fingers
        pinky_flex  = int(hand_physics.get_finger_flexion(hand_world_landmarks, 4))
        ring_flex   = int(hand_physics.get_finger_flexion(hand_world_landmarks, 3))
        middle_flex = int(hand_physics.get_finger_flexion(hand_world_landmarks, 2))
        index_flex  = int(hand_physics.get_finger_flexion(hand_world_landmarks, 1))
        thumb_flex  = int(hand_physics.get_finger_flexion(hand_world_landmarks, 0))

        # Additional metrics
        thumb_palm = int(hand_physics.calculate_thumb_opposition(hand_world_landmarks))
        _, wrist_yaw = hand_physics.calculate_wrist_angles(hand_world_landmarks)

        # Map to 0-180 payload
        payload = [
            max(0, min(180, pinky_flex)),
            max(0, min(180, ring_flex)),
            max(0, min(180, middle_flex)),
            max(0, min(180, index_flex)),
            max(0, min(180, thumb_palm)),
            max(0, min(180, thumb_flex)),
            max(0, min(180, int(wrist_yaw))),
            max(0, min(180, int(elbow_angle)))
        ]

        checksum = self.calculate_checksum(payload)
        
        # Pack into 10 bytes: Header + 8 payload bytes + Checksum
        try:
            return struct.pack(self.packet_format, self.HEADER_TX, *payload, checksum)
        except Exception as e:
            print(f"Packet packing error: {e}")
            return None

    def decode_fpga_packet(self, packet: bytes) -> dict:
        """
        Decodes a 10-byte packet received from the FPGA.
        Returns a dictionary of servo angles if the checksum is valid.
        """
        if not packet or len(packet) != 10:
            return None

        try:
            unpacked = struct.unpack(self.packet_format, packet)
            header = unpacked[0]
            payload = unpacked[1:9]
            checksum = unpacked[9]

            if header != self.HEADER_RX:
                return None

            # Verify checksum
            if self.calculate_checksum(payload) != checksum:
                return None

            return {
                "pinky":      payload[0],
                "ring":       payload[1],
                "middle":     payload[2],
                "index":      payload[3],
                "thumb_palm": payload[4],
                "thumb":      payload[5],
                "wrist":      payload[6],
                "elbow":      payload[7]
            }
        except Exception:
            return None
