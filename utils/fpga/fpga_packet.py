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

    # --- Finger flexion calibration -------------------------------------
    # The raw PIP-joint flexion (180 - interior angle) never reaches 180 even
    # when a finger is fully curled: it spans roughly FLEX_MIN (straight) to
    # FLEX_MAX (fully closed). We stretch that usable band onto the servo's
    # full 0-180 travel so a closed fist drives the servos all the way down.
    # Tune FLEX_MIN/FLEX_MAX to your own hand if the ends clip or fall short.
    FLEX_MIN = 15.0    # deg of flexion with the finger held straight
    FLEX_MAX = 110.0   # deg of flexion with the finger fully curled

    # --- Thumb opposition (inner palm) calibration ----------------------
    # calculate_thumb_opposition returns a distance (thumb tip -> pinky base):
    # large when the thumb is open, ~0 when closed onto the palm. We map that
    # distance band to 0-180 and invert it so "closed" = large command.
    OPP_MIN = 20.0     # scaled distance with thumb closed onto the palm
    OPP_MAX = 160.0    # scaled distance with thumb fully abducted

    def __init__(self):
        # Format: 1 byte header, 8 bytes payload (servos), 1 byte checksum
        self.packet_format = ">B8BB"

    @staticmethod
    def _scale(value, lo, hi, invert=False):
        """Linearly map value from [lo, hi] onto servo range [0, 180].

        Set invert=True when a small input should mean a large servo command
        (used for the elbow, wrist and thumb-opposition channels whose motion
        is otherwise reversed relative to the intended movement).
        """
        if hi == lo:
            return 0
        norm = (value - lo) / (hi - lo)
        norm = max(0.0, min(1.0, norm))
        if invert:
            norm = 1.0 - norm
        return int(round(norm * 180))

    def calculate_checksum(self, header, payload):
        """Calculates XOR checksum over the header byte and all payload bytes.

        The VHDL frame_rx seeds its XOR accumulator with the direction byte
        (0xFF/0xFE) before folding in the data bytes, so the checksum must
        include the header to match (frame_rx.vhd lines 55, 67, 77).
        """
        checksum = header
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

        # Map to 0-180 servo payload.
        #  - Fingers: stretch the usable flexion band onto full servo travel so
        #    a closed fist reaches the servo's end stop instead of stopping short.
        #  - Thumb-palm / wrist / elbow: invert so the servo follows the intended
        #    direction of movement (see calibration notes above).
        payload = [
            self._scale(pinky_flex,  self.FLEX_MIN, self.FLEX_MAX),
            self._scale(ring_flex,   self.FLEX_MIN, self.FLEX_MAX),
            self._scale(middle_flex, self.FLEX_MIN, self.FLEX_MAX),
            self._scale(index_flex,  self.FLEX_MIN, self.FLEX_MAX),
            self._scale(thumb_palm,  self.OPP_MIN,  self.OPP_MAX, invert=True),
            self._scale(thumb_flex,  self.FLEX_MIN, self.FLEX_MAX),
            max(0, min(180, 180 - int(wrist_yaw))),
            max(0, min(180, 180 - int(elbow_angle)))
        ]

        checksum = self.calculate_checksum(self.HEADER_TX, payload)

        # Pack into 10 bytes: Header + 8 payload bytes + Checksum
        try:
            return struct.pack(self.packet_format, self.HEADER_TX, *payload, checksum)
        except Exception as e:
            print(f"Packet packing error: {e}")
            return None

    def create_read_request(self, last_packet: bytes = None) -> bytes:
        """
        Builds a 10-byte read-request frame (header 0xFE).

        The FPGA (basys3_top.vhd) only starts a readback when it receives a
        frame whose direction byte is 0xFE (frame_valid + frame_dir = x"FE").

        IMPORTANT: frame_rx accepts 0xFF and 0xFE identically, and the current
        bitstream writes frame_angles to the servo mailbox for ANY valid frame
        (mb_wr_valid = frame_valid and not chksum_err). So a 0xFE frame with a
        zero payload would clobber the servo command to zero every frame and
        freeze the hand. To stay safe on the un-reflashed bitstream we echo the
        last command's payload here, so the (harmless) double-write repeats the
        same angles instead of zeroing them. Once the VHDL gates the mailbox on
        frame_dir = 0xFF, the payload no longer matters.
        """
        if last_packet and len(last_packet) >= 9:
            payload = list(last_packet[1:9])
        else:
            payload = [0] * 8
        checksum = self.calculate_checksum(self.HEADER_RX, payload)
        return struct.pack(self.packet_format, self.HEADER_RX, *payload, checksum)

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

            # Verify checksum (VHDL seeds the XOR with the direction byte)
            if self.calculate_checksum(self.HEADER_RX, payload) != checksum:
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
