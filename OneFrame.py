import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Main loop
while True:
    _, frame = cap.read()
    # Processing
    frame = cv2.resize(frame, (4, 5))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame[0, 0] = 255

    # print("Shape: ", frame.shape)
    # print("ndim: ", frame.ndim)
    # print("dtype: ", frame.dtype)
    print(frame)
    break

cap.release()
cv2.destroyAllWindows()
