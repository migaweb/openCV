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


class GameObject:
    def __init__(self, width, height, size=50, object_image='logo.png'):
        self.width = width
        self.height = height
        self.size = size
        self.x = np.random.randint(0, self.width - self.size)
        self.y = np.random.randint(0, self.height - self.size)
        self.logo = cv2.imread(object_image)
        self.logo = cv2.resize(self.logo, (self.size, self.size))
        gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    def update_frame(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.mask)] = 0
        roi += self.logo


class PlayerObject(GameObject):
    def update_position(self, fg_mask):
        roi = fg_mask[self.y:self.y + self.size, self.x: self.x + self.size]
        check_hit = np.any(roi[np.where(self.mask)])
        if check_hit:
            best_fit = np.inf
            best_delta_x = 0
            best_delta_y = 0
            for _ in range(8):
                delta_x = np.random.randint(-15, 16)
                delta_y = np.random.randint(-15, 16)
                if self.x + delta_x < 0 or self.x + self.size + delta_x >= self.width or self.y + delta_y < 0 or self.y + self.size + delta_y >= self.height:
                    continue
                roi = fg_mask[self.y + delta_y:self.y + delta_y + self.size, self.x + delta_x: self.x + delta_x + self.size]
                overlap = np.count_nonzero(roi[np.where(self.mask)])

                if overlap < best_fit:
                    best_fit = overlap
                    best_delta_x = delta_x
                    best_delta_y = delta_y

            self.x += best_delta_x
            self.y += best_delta_y
        return check_hit


class TrackerObject(GameObject):
    def __init__(self, width, height, follow=None):
        GameObject.__init__(self, width, height, size=25, object_image='logo_red.png')
        self.speed_x = 0
        self.speed_y = 0
        self.max_speed = 2
        self.follow = follow

    def update_position(self):
        diff_x = self.follow.x + self.follow.size//2 - (self.x + self.size//2)
        diff_y = self.follow.y + self.follow.size//2 - (self.y + self.size//2)

        if abs(diff_x) < self.size and abs(diff_y) < self.size:
            return True

        if diff_x < 0 and not self.speed_x < -self.max_speed:
            self.speed_x -= np.random.randint(0, 2)
        elif diff_x > 0 and not self.speed_x > self.max_speed:
            self.speed_x += np.random.randint(0, 2)
        if diff_y < 0 and not self.speed_y < -self.max_speed:
            self.speed_y -= np.random.randint(0, 2)
        elif diff_y > 0 and not self.speed_y > self.max_speed:
            self.speed_y += np.random.randint(0, 2)

        if self.x + self.speed_x < 0 or self.x + self.speed_x + self.size >= self.width or \
                self.y + self.speed_y < 0 or self.y + self.speed_y + self.size >= self.height:
            return False

        self.x += self.speed_x
        self.y += self.speed_y
        return False


class Game:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.player = PlayerObject(width, height)
        self.goal = GameObject(width, height, object_image='logo_gray.png')
        self.trackers = []
        self.trackers.append(TrackerObject(width, height, follow=self.player))
        self.trackers.append(TrackerObject(width, height, follow=self.player))
        self.score = 0
        self.hit = False

    def update_position(self, fg_mask):
        self.player.update_position(fg_mask)
        self.hit = False
        for tracker in self.trackers:
            result = tracker.update_position()
            if result:
                self.hit = True

    def update_frame(self, frame):
        self.goal.update_frame(frame)
        self.player.update_frame(frame)
        for tracker in self.trackers:
            tracker.update_frame(frame)

        if abs(self.player.x + self.player.size//2 - (self.goal.x + self.goal.size//2)) < self.player.size//2 and \
                abs(self.player.y + self.player.size//2 - (self.goal.y + self.goal.size//2)) < self.player.size//2:
            self.score += 1
            self.goal.x = np.random.randint(0, self.width - self.goal.size)
            self.goal.y = np.random.randint(0, self.height - self.goal.size)
            frame[:, :, 2] = 255

        if self.hit:
            self.score -= 1
            self.player.x = np.random.randint(0, self.width-self.player.size)
            self.player.y = np.random.randint(0, self.height-self.player.size)
            frame[:, :, 2] = 255


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

        game.update_position(fg_mask)
        game.update_frame(frame)

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
