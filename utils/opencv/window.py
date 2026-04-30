import cv2

class OpenCVWindow:
    """
    A helper class to manage an OpenCV display window.
    It handles window creation, frame display, and provides a clean interface
    for checking if the window should be closed (via key press or 'X' button).
    """
    def __init__(self, window_name="Hand Detection"):
        """
        Initializes the OpenCV window with a specific name.
        """
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show(self, frame):
        """
        Displays the provided image frame in the window.
        """
        cv2.imshow(self.window_name, frame)

    def should_close(self):
        """
        Checks for exit conditions:
        1. User pressed 'Esc' (ASCII 27).
        2. User pressed 'q' (ASCII 113).
        3. User clicked the 'X' button to close the window.
        Returns:
            bool: True if any exit condition is met, False otherwise.
        """
        # Check if 'Esc' (27) or 'q' (113) was pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            return True

        # Check if the 'X' button was clicked (window property becomes < 1)
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        return False

    def close(self):
        """
        Destroys the OpenCV window and releases associated resources.
        """
        cv2.destroyWindow(self.window_name)

