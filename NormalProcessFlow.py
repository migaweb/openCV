import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:

    # Main loop
    while True:
        _, frame = cap.read()
        # Processing
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
