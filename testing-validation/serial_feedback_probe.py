"""
Standalone serial feedback probe.

Purpose: determine — with zero MediaPipe/camera involvement — whether the FPGA
actually transmits anything back over UART when asked.

It does three things in a loop:
  1. sends a 0xFF write frame (neutral 90-degree pose, so servos hold still),
  2. sends a 0xFE read-request frame (the only thing that triggers readback),
  3. dumps EVERY byte that arrives, as hex, with a decode attempt.

Run it AFTER flashing the updated bitstream:
    python serial_feedback_probe.py

Interpreting the output:
  - "RX raw: <hex>"  -> the FPGA IS sending. If it starts with fe and is 10
                        bytes, feedback works; we just decode it.
  - "RX raw: (nothing)" every line -> the FPGA sends NOTHING. The problem is on
                        the FPGA/UART side (bitstream not flashed, frame_tx not
                        triggered, or TX pin), not in the main Python app.
"""
import sys
import os
import time
import serial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.fpga.fpga_packet import FPGAPacket

PORT = 'COM16'
BAUD = 115200

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.3)
    except Exception as e:
        print(f"Could not open {PORT}: {e}")
        return

    print(f"Opened {PORT} @ {BAUD}. Probing feedback (Ctrl+C to stop)...\n")
    pkt = FPGAPacket()

    # Neutral 90-degree command so the servos don't thrash during the probe.
    neutral = [90] * 8
    cmd_checksum = pkt.calculate_checksum(FPGAPacket.HEADER_TX, neutral)
    import struct
    cmd = struct.pack(pkt.packet_format, FPGAPacket.HEADER_TX, *neutral, cmd_checksum)
    req = pkt.create_read_request(cmd)

    print(f"TX command : {cmd.hex()}")
    print(f"TX request : {req.hex()}\n")

    try:
        n = 0
        while True:
            n += 1
            ser.reset_input_buffer()
            ser.write(cmd)      # write frame (0xFF)
            ser.write(req)      # read-request frame (0xFE)
            ser.flush()

            time.sleep(0.05)    # give the FPGA time to respond
            waiting = ser.in_waiting
            data = ser.read(max(waiting, 20))

            if data:
                decoded = pkt.decode_fpga_packet(data[:10]) if len(data) >= 10 else None
                print(f"[{n:04d}] in_waiting={waiting:3d}  RX raw: {data.hex()}"
                      + (f"  decoded={decoded}" if decoded else "  decoded=None"))
            else:
                print(f"[{n:04d}] in_waiting={waiting:3d}  RX raw: (nothing)")

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
