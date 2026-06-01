import numpy as np
import cv2 as cv

def csp(size):
    blkimg = np.zeros((size, size), dtype=np.uint8)
    cv.imshow("First image", blkimg)

csp(512)
cv.waitKey(0)