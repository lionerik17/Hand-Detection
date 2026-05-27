"""
Module for constructing and decoding binary packets for FPGA communication.
11-byte UART frames: direction + 9 servo angles + XOR checksum.
Must match frame_rx.vhd / frame_tx.vhd protocol on the Basys3.
"""

import struct
from ..hand import hand_physics

HEADER_TX = 0xFF  # Write servo angles
HEADER_RX = 0xFE  # Read-back from FPGA
PACKET_FMT = '!BBBBBBBBBBB'

def _clamp_angle(value):
    return max(0, min(180, int(value)))

def create_fpga_packet(world_landmarks, elbow_angle=0):
    """
    Constructs an 11-byte packet matching the VHDL frame_rx protocol.

    Servo mapping:
      [1] Pinky, [2] Ring, [3] Middle, [4] Index,
      [5] Thumb palm, [6] Thumb tension, [7] Wrist,
      [8] Elbow, [9] Thumb torsion

    Checksum: XOR of bytes 0-9 (receiver XORs all 11 bytes, expects 0x00).
    """
    if not world_landmarks:
        return None

    flexions = [int(hand_physics.get_finger_flexion(world_landmarks, i)) for i in range(5)]
    pitch, yaw = hand_physics.calculate_wrist_angles(world_landmarks)
    opposition = hand_physics.calculate_thumb_opposition(world_landmarks)

    payload = [
        _clamp_angle(flexions[4]),     # Servo 0: Pinky
        _clamp_angle(flexions[3]),     # Servo 1: Ring
        _clamp_angle(flexions[2]),     # Servo 2: Middle
        _clamp_angle(flexions[1]),     # Servo 3: Index
        _clamp_angle(opposition),      # Servo 4: Thumb palm
        _clamp_angle(flexions[0]),     # Servo 5: Thumb tension
        _clamp_angle(pitch),           # Servo 6: Wrist
        _clamp_angle(elbow_angle),     # Servo 7: Elbow
        _clamp_angle(opposition),      # Servo 8: Thumb torsion
    ]

    checksum = HEADER_TX
    for b in payload:
        checksum ^= b

    packet_data = [HEADER_TX] + payload + [checksum]
    return struct.pack(PACKET_FMT, *packet_data)

def decode_fpga_packet(packet_bytes):
    """
    Decodes an 11-byte response from the FPGA (header 0xFE).
    Validates XOR checksum: XOR of all 11 bytes must equal 0x00.
    """
    if not packet_bytes or len(packet_bytes) != 11:
        return None

    values = struct.unpack(PACKET_FMT, packet_bytes)

    if values[0] != HEADER_RX:
        return None

    xor_check = 0
    for b in values:
        xor_check ^= b
    if xor_check != 0x00:
        print(f"Error: XOR checksum failed (got 0x{xor_check:02X}, expected 0x00)")
        return None

    return {
        "pinky":         values[1],
        "ring":          values[2],
        "middle":        values[3],
        "index":         values[4],
        "thumb_palm":    values[5],
        "thumb":         values[6],
        "wrist":         values[7],
        "elbow":         values[8],
        "thumb_torsion": values[9],
    }
