import cv2
import numpy as np

## task 1
img = cv2.imread('text.bmp', cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (41, 23), 0)
img = cv2.Laplacian(img, ddepth = cv2.CV_32F, ksize=11)
(thresh, img) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite('task1out.bmp', img)

## task 2
# transform words to shapes using dilate / erode
img = 255 - cv2.imread('text.bmp', cv2.IMREAD_GRAYSCALE)
img = cv2.dilate(img, np.ones((4, 5), np.uint8), iterations=1)
img = cv2.erode(img, np.ones((4, 4), np.uint8), iterations=1)
img = cv2.dilate(img, np.ones((1, 5), np.uint8), iterations=1)
img = cv2.erode(img, np.ones((4, 4), np.uint8), iterations=1)
img = cv2.dilate(img, np.ones((1, 5), np.uint8), iterations=1)
img = cv2.erode(img, np.ones((4, 4), np.uint8), iterations=1)
img = cv2.dilate(img, np.ones((8, 1), np.uint8), iterations=1)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite('task2out1.bmp', img)

# find enclosing rectangles using floodFill
(height, width) = img.shape
msk = np.zeros((height + 2, width + 2), np.uint8)
rects = []
for i in range(0, width):
    for j in range(0, height):
        (success, rect) = cv2.floodFill(img, seedPoint=(i, j), newVal=255, mask=msk, flags=cv2.FLOODFILL_MASK_ONLY)
        if success != 0:
            rects.append(rect)

# draw found rects on image and save
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for r in rects[1:]:
    cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)
cv2.imwrite('task2out2.bmp', img)