import cv2
import time

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 10)
print("FPS: ", cap.get(cv2.CAP_PROP_FPS))

last_time = time.time()

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))
    frame = cv2.flip(frame, 1)

    try:
        text = "FPS: " + str(int(1/(time.time()-last_time)))
        last_time = time.time()
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    except:
        cv2.putText(frame, "FPS: N/A", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
