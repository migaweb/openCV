import cv2
import numpy as np

size = 100
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

logo = cv2.imread('logo.png')
logo = cv2.resize(logo, (size, size))
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

try:

    # Main loop
    while True:
        _, frame = cap.read()
        # Processing
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        roi = frame[10:size+10, 10:size+10]
        roi[np.where(mask)] = 0
        roi += logo

        # Show image in window
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
