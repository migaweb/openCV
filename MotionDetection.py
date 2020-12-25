import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    for _ in range(10):
        _, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    background = gray

    # cv2.imshow("Background", background)
    last_frame = gray

    # Main loop
    while True:
        _, frame = cap.read()
        # Reading, resizing and flipping frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (17, 17), 0)

        # Processing

        # Noise problem solving using abs diff
        abs_diff = cv2.absdiff(last_frame, gray)
        last_frame = gray
        _, ad_mask = cv2.threshold(abs_diff, 15, 255, cv2.THRESH_BINARY)
        # dilated_mask = cv2.dilate(ad_mask, None, iterations=2)

        contours, _ = cv2.findContours(ad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # avoid small movement
            if cv2.contourArea(contour) < 1000:
                continue
            # This means movement detected
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show image in window
        cv2.imshow("Mask", ad_mask)
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
