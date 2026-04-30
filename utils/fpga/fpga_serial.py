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

    def receive_packet(self):
        """
        Reads a 10-byte response from the device if enough data is available in the buffer.
        Returns:
            bytes or None: The 10-byte packet if received, None otherwise.
        """
        if self.is_connected() and self.ser.in_waiting >= 10:
            return self.ser.read(10)
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
