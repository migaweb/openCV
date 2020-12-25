import cv2
import numpy as np
from collections import deque


class BackgroundExtraction:
    def __init__(self, width, height, scale, max_len=10):
        self.max_len = max_len
        self.width = width//scale
        self.height = height//scale
        self.buffer = deque(maxlen=max_len)
        self.background = None

    def calculata_background(self):
        self.background = np.zeros((self.height, self.width), dtype='float32')
        for item in self.buffer:
            self.background += item
        self.background /= len(self.buffer)

    def update_background(self, old_frame, new_frame):
        self.background -= old_frame/self.max_len
        self.background += new_frame/self.max_len

    def updateFrame(self, frame):
        if len(self.buffer) < self.max_len:
            self.buffer.append(frame)
            self.calculata_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_background(self):
        return self.background.astype('uint8')


width = 640
height = 480
scale = 2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bg_buffer = BackgroundExtraction(width, height, scale, 30)

try:
    # Main loop
    while True:
        _, frame = cap.read()
        # Reading, resizing and flipping frame
        frame = cv2.resize(frame, (width, height))
        frame = cv2.flip(frame, 1)

        # processing the frame
        down_scale = cv2.resize(frame, (width//scale, height//scale))
        gray = cv2.cvtColor(down_scale, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Noise problem solving using abs diff
        bg_buffer.updateFrame(gray)
        abs_diff = cv2.absdiff(bg_buffer.get_background(), gray)
        _, ad_mask = cv2.threshold(abs_diff, 15, 255, cv2.THRESH_BINARY)
        # dilated_mask = cv2.dilate(ad_mask, None, iterations=2)

        contours, _ = cv2.findContours(ad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # avoid small movement
            if cv2.contourArea(contour) < 250:
                continue
            # This means movement detected
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = x*scale, y*scale, w*scale, h*scale
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show image in window
        cv2.imshow("Webcam", frame)
        # cv2.imshow("DilatedMask", dilated_mask)

        # Q key terminates
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

except Exception as e:
    print(e)
    cap.release()
    cv2.destroyAllWindows()
