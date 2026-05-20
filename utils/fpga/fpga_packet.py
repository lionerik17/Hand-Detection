"""
Module for constructing and decoding binary packets for FPGA communication.
Implements a 10-byte protocol with headers, payload metrics, and checksums.
"""

import struct
from ..hand import hand_physics

HEADER_TX = 0xFF  # Header for packets sent TO FPGA
HEADER_RX = 0xFE  # Header for packets received FROM FPGA

# Wrist Multiplexing Modes
WRIST_MODE_PITCH = 0
WRIST_MODE_YAW = 1

def create_fpga_packet(world_landmarks, wrist_mode=WRIST_MODE_PITCH):
    """
    Constructs a 10-byte binary packet from hand world landmarks.

    Structure:
    [0] Header (0xFF)
    [1-5] Finger Flexions (Thumb, Index, Middle, Ring, Pinky)
    [6] Thumb Opposition
    [7] Finger Spread
    [8] Wrist (Multiplexed: 0-127 Pitch, 128-255 Yaw)
    [9] Checksum (8-bit sum of bytes 1-8 mod 256)

    Args:
        world_landmarks (list): MediaPipe hand world landmarks.
        wrist_mode (int): WRIST_MODE_PITCH or WRIST_MODE_YAW.

    Returns:
        bytes or None: 10-byte binary packet ready for serial transmission.
    """
    if not world_landmarks:
        return None
    
    # Calculate finger flexions
    flexions = [int(hand_physics.get_finger_flexion(world_landmarks, i)) for i in range(5)]

    # Calculate specialized metrics
    opposition = hand_physics.calculate_thumb_opposition(world_landmarks)
    spread = hand_physics.calculate_finger_spread(world_landmarks)
    pitch, yaw = hand_physics.calculate_wrist_angles(world_landmarks)

    # Multiplex Wrist: Pitch or Yaw based on mode
    # Clamp both to 0-127 range before applying offset
    if wrist_mode == WRIST_MODE_PITCH:
        wrist_byte = max(0, min(127, pitch))
    else:
        wrist_byte = 128 + max(0, min(127, yaw))

    # Assemble Payload (Bytes 1-8)
    payload = [
        max(0, min(255, flexions[0])), # Thumb
        max(0, min(255, flexions[1])), # Index
        max(0, min(255, flexions[2])), # Middle
        max(0, min(255, flexions[3])), # Ring
        max(0, min(255, flexions[4])), # Pinky
        max(0, min(255, opposition)),
        max(0, min(255, spread)),
        wrist_byte
    ]

    # Calculate checksum (8-bit sum of payload)
    checksum = sum(payload) % 256
    
    # Build final 10-byte packet
    packet_data = [HEADER_TX] + payload + [checksum]
    return struct.pack('!BBBBBBBBBB', *packet_data)

def decode_fpga_packet(packet_bytes):
    """
    Decodes a 10-byte binary packet received from the FPGA.

    Args:
        packet_bytes (bytes): 10-byte raw binary data.

    Returns:
        dict or None: Decoded metrics if validation (header/checksum) passes.
    """
    if not packet_bytes or len(packet_bytes) != 10:
        return None

    # Unpack binary data: 10 unsigned bytes
    values = struct.unpack('!BBBBBBBBBB', packet_bytes)

    header = values[0]
    payload = values[1:9]
    checksum = values[9]

    # Validation
    if header != HEADER_RX:
        return None

    if sum(payload) % 256 != checksum:
        print(f"Error: Checksum mismatch. Expected {checksum}, got {sum(payload)%256}")
        return None
    
    # Return mapped dictionary
    # Wrist is multiplexed: we return both keys but only one is 'new'
    wrist_val = payload[7]
    res = {
        "thumb":      payload[0],
        "index":      payload[1],
        "middle":     payload[2],
        "ring":       payload[3],
        "pinky":      payload[4],
        "opposition": payload[5],
        "spread":     payload[6],
    }

    if wrist_val < 128:
        res["wrist_pitch"] = wrist_val
        res["wrist_yaw"] = None # Signal that this frame didn't update yaw
    else:
        res["wrist_pitch"] = None # Signal that this frame didn't update pitch
        res["wrist_yaw"] = wrist_val - 128
    
    return res
