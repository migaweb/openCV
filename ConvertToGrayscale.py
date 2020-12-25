import cv2
import numpy as np

image = np.array([[[0, 0, 0], [255, 255, 255], [127, 127, 127]],
                  [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
                  [[0, 255, 255], [255, 255, 0], [255, 0, 255]]], dtype="uint8")

# print(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
image = cv2.resize(image, (50, 50))
gray = cv2.resize(gray, (50, 50))
cv2.imshow("Image", image)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
