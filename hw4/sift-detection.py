import cv2
import numpy as np
from matplotlib import pyplot as plt

MAX_DIST = 1
MAX_FEATURES = 1000

def drawMatches(img1, kp1, img2, kp2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    out_img = np.zeros((max(rows1, rows2), cols1 + cols2, 3), np.uint8)
    out_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    out_img[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])

    for match in matches:
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out_img, (int(x1), int(y1)), 4, (0, 0, 255), 2)
        cv2.circle(out_img, (int(x2) + cols1, int(y2)), 4, (0, 0, 255), 1)

        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)

    return out_img

# main
# read img
img = cv2.imread('mandril.bmp')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# get transformed img
rows, cols = img.shape
transMatrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
imgTransformed = cv2.warpAffine(img, transMatrix, (cols,rows))

# find the keypoints and descriptors with SIFT
siftDetector = cv2.SIFT(MAX_FEATURES)
srcKpts, srcDescrs = siftDetector.detectAndCompute(img, None)
transfKpts, transfDescrs = siftDetector.detectAndCompute(imgTransformed, None)

# save
imgWithKpts = cv2.drawKeypoints(img, srcKpts)
cv2.imwrite('original.bmp', imgWithKpts)
imgWithKpts = cv2.drawKeypoints(imgTransformed, transfKpts)
cv2.imwrite('transformed.bmp', imgWithKpts)
imgWithKpts = cv2.drawKeypoints(img, srcKpts, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('originalWithSizes.bmp', imgWithKpts)
imgWithKpts = cv2.drawKeypoints(imgTransformed, transfKpts, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('transformedWithSizes.bmp', imgWithKpts)

# get matches with flann
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks = 50)   # or pass empty dictionary
flannMatcher = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flannMatcher.knnMatch(srcDescrs, transfDescrs, k = 1)

# expected positions
homogSrcPts = map(lambda kp: [kp.pt[0], kp.pt[1], 1], srcKpts)
expectedPts = map(lambda hp: np.dot(transMatrix, hp), homogSrcPts)

# count actual matches (dist < MAX_DIST)
actualMatches = []
for m in matches:
    if len(m) == 0:
        continue
    actual = transfKpts[m[0].trainIdx]
    expected = expectedPts[m[0].queryIdx]
    dist = np.linalg.norm(np.array(actual.pt) - np.array(expected))
    if dist < MAX_DIST:
        actualMatches.append(m)

# results
matchesImg = drawMatches(img, srcKpts, imgTransformed, transfKpts, actualMatches)
cv2.imwrite('goodMatches.bmp', matchesImg)

print "all matches: " + str(len(matches))
print "matches where dist < " + str(MAX_DIST) + ": " + str(len(actualMatches))
print "percentage: " + str((1.0 * len(actualMatches)) / len(matches) * 100)