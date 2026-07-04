import serial

class FPGASerial:
    """
    A class to handle serial communication with an external FPGA device.
    It encapsulates port initialization, connection management, and data transfer logic.
    """
    def __init__(self, port='COM3', baudrate=115200, timeout=0.1):
        """
        Initializes the serial configuration and attempts an initial connection.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connect()

    def connect(self):
        """
        Attempts to open the serial port using the current configuration.
        Prints a connection message on success or a warning on failure.
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Connected to FPGA on {self.port}")
        except Exception as e:
            print(f"Warning: Could not open serial port {self.port}: {e}")
            self.ser = None

    def send_packet(self, packet_bytes):
        """
        Writes raw bytes to the serial port if a connection is active.
        Args:
            packet_bytes (bytes): The data packet to transmit.
        Returns:
            bool: True if transmission was successful, False otherwise.
        """
        if self.is_connected() and packet_bytes:
            self.ser.write(packet_bytes)
            return True
        return False

    def receive_packet(self, header=0xFE, length=10):
        """
        Reads one framed response from the device, synchronising on the header.

        Unlike a bare read(10), this scans for the direction byte (0xFE) and
        then reads the remaining bytes, so it stays aligned even if the serial
        buffer starts mid-frame. Bounded by the port timeout, so it will not
        block forever when the FPGA sends nothing.

        Returns:
            bytes or None: The `length`-byte packet starting at `header`, or None.
        """
        if not self.is_connected():
            return None

        # Find the header byte (read one byte at a time, limited by timeout).
        # Give up after scanning a couple of frame-lengths worth of bytes so a
        # silent FPGA doesn't stall the capture loop.
        for _ in range(length * 3):
            b = self.ser.read(1)
            if not b:            # timeout: nothing arriving
                return None
            if b[0] == header:
                rest = self.ser.read(length - 1)
                if len(rest) == length - 1:
                    return bytes([header]) + rest
                return None
        return None

    def is_connected(self):
        """
        Verifies if the serial port object exists and is currently open.
        Returns:
            bool: True if connected, False otherwise.
        """
        return self.ser is not None and self.ser.is_open

    def close(self):
        """
        Safely closes the serial port and clears the serial object.
        """
        if self.ser:
            self.ser.close()
            self.ser = None
