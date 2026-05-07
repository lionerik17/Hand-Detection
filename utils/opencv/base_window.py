import cv2

class OpenCVWindow:
    """
    Base class for managing an OpenCV display window.
    """
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def should_close(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            return True
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except:
            return True
        return False

    def close(self):
        cv2.destroyWindow(self.window_name)
