import cv2

logo = cv2.imread('logo.png')
logo = cv2.resize(logo, (10, 10))
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
roi = gray[3:6, 3:6]
roi[:, :] = gray[0:3, 0:3]
print(gray)
print("roi: ", roi)

cv2.imshow("View", gray)
cv2.waitKey(0)
