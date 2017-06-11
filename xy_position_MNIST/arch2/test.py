import cv2

image = cv2.imread("./testimages/30.jpg")

cv2.imshow("test", cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA))

cv2.imshow("resized test", cv2.resize(cv2.resize(image, (14, 14), interpolation= cv2.INTER_CUBIC), (224, 224), interpolation=cv2.INTER_AREA))

cv2.waitKey()