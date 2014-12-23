import numpy as np
import cv2

def featurePointsShiTomasi(img):
    feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    return cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

def featurePointsHarris(img):
    img = np.float32(img)
    cornerMap = cv2.cornerHarris(img, 2, 3, 0.04)
    detected = []
    maxResponse = cornerMap.max()
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if cornerMap[y][x] > 0.1 * maxResponse:
                # cv2.circle(img, (x, y), 2, (155, 0, 25))
                detected.append([x, y])
    return np.asarray(detected, dtype=np.float32).reshape(-1, 1, 2)

def featurePointsFAST(img):
    fast = cv2.FastFeatureDetector(threshold=120)
    fast.setBool('nonmaxSuppression',0)
    kp = fast.detect(img,None)
    pts = map(lambda p: (p.pt), kp)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

cap = cv2.VideoCapture('sequence.mpg')

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
cv2.imwrite('firstFrame.bmp', old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# p0 = featurePointsShiTomasi(old_gray)
# p0 = featurePointsHarris(old_gray)
p0 = featurePointsFAST(old_gray)

# Create some random colors
color = np.random.randint(0, 255, (len(p0), 3))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1)
        cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
