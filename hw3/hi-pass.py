import cv2
import numpy as np
# from matplotlib import pyplot as plt

NULL_RADIUS = 30

img = cv2.imread('mandril.bmp', cv2.IMREAD_GRAYSCALE)
# fft
img_transformed = np.fft.fft2(img)
img_transshifted = np.fft.fftshift(img_transformed)
# null low freqs
rows, cols = img.shape
crow, ccol = rows / 2 , cols / 2
img_transshifted[crow - NULL_RADIUS : crow + NULL_RADIUS, ccol - NULL_RADIUS : ccol + NULL_RADIUS] = 0
# ifft
img_transshifted_inv = np.fft.ifftshift(img_transshifted)
img_restored = np.fft.ifft2(img_transshifted_inv)
img_restored = np.abs(img_restored)
cv2.imwrite('fftOut.bmp', np.abs(img_restored))

# now Laplacian to compare
img_lap = cv2.Laplacian(img, ddepth = cv2.CV_32F, ksize=1)
cv2.imwrite('laplacianOut.bmp', np.abs(img_lap))