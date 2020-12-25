import cv2

logo = cv2.imread('logo.png')
print(logo.shape)

logo = cv2.resize(logo, (10, 10))
logo_default = cv2.resize(logo, (150, 150))

cv2.imshow("DefaultLogo", logo_default)

logo_cubic = cv2.resize(logo, (150, 150), interpolation=cv2.INTER_CUBIC)
cv2.imshow("CubicLogo", logo_cubic)

cv2.waitKey(0)