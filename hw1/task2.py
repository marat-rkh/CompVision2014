import cv2
import numpy as np

# task 1
img = cv2.imread('text.bmp', cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (21, 23), 0)
img = cv2.Laplacian(img, cv2.CV_32F, ksize=11)
cv2.imwrite('task1out.bmp', img)

# task 2
eKer = np.ones((5, 5), np.uint8)
img = cv2.erode(img, eKer, iterations=1)
dKer = np.ones((8, 7), np.uint8)
img = cv2.dilate(img, dKer, iterations=1)
eKer = np.ones((3, 3), np.uint8)
img = cv2.erode(img, eKer, iterations=1)
cv2.imwrite('task2out1.bmp', img)

ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite('task2out2.bmp', img)