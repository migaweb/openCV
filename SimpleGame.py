import cv2
import numpy as np
from collections import deque


class BackgroundExtraction:
    def __init__(self, width, height, scale, max_len=10):
        self.max_len = max_len
        self.scale = scale
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

    def update_frame(self, frame):
        if len(self.buffer) < self.max_len:
            self.buffer.append(frame)
            self.calculata_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_background(self):
        return self.background.astype('uint8')

    def apply(self, frame):
        # processing the frame
        down_scale = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(down_scale, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Noise problem solving using abs diff
        self.update_frame(gray)
        abs_diff = cv2.absdiff(bg_buffer.get_background(), gray)
        _, ad_mask = cv2.threshold(abs_diff, 15, 255, cv2.THRESH_BINARY)
        # dilated_mask = cv2.dilate(ad_mask, None, iterations=2)
        return cv2.resize(ad_mask, (self.width*self.scale, self.height*self.scale))


class Game:
    def __init__(self, width, height, size=50):
        self.score = 0
        self.width = width
        self.height = height
        self.size = size
        self.x = np.random.randint(0, self.width - self.size)
        self.y = 0
        self.speed = 10
        self.logo = cv2.imread('logo.png')
        self.logo = cv2.resize(self.logo, (self.size, self.size))
        gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    def update_frame(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.mask)] = 0
        roi += self.logo

    def update_position(self, fg_mask):
        self.y += self.speed

        if self.y + self.size >= self.height:
            self.score -= 1
            self.y = 0
            self.speed = np.random.randint(10, 20)
            self.x = np.random.randint(0, self.width - self.size)

        roi = fg_mask[self.y:self.y + self.size, self.x: self.x + self.size]
        check_hit = np.any(roi[np.where(self.mask)])
        if check_hit:
            self.score += 1
            self.y = 0
            self.speed = np.random.randint(10, 25)
            self.x = np.random.randint(0, self.width - self.size)

        return check_hit


width = 1280
height = 720
scale = 2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bg_buffer = BackgroundExtraction(width, height, scale, 5)
game = Game(width, height)
try:
    # Main loop
    while True:
        _, frame = cap.read()
        # Reading, resizing and flipping frame
        frame = cv2.resize(frame, (width, height))
        frame = cv2.flip(frame, 1)

        fg_mask = bg_buffer.apply(frame)

        hit = game.update_position(fg_mask)
        game.update_frame(frame)

        if hit:
            frame[:, :, 2] = 255

        text = f"Score: {game.score}"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

        # Show image in window
        # cv2.imshow("FG Mask", fg_mask)
        cv2.imshow("Webcam", frame)

        # Q key terminates
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

except Exception as e:
    print(e)
    cap.release()
    cv2.destroyAllWindows()
