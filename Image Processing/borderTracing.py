# ============================================================================ #
# Contains the functions for tracing out a border, of a specific color for an image.
# Generates an svg path connecting the pixels

import numpy as np
import cv2 as cv
import ColorPalettes

IMAGE = cv.imread("Image Processing/filteredImage.png")

def padImage(image):
    """
    Takes a binary mask and pads all edges with 0's for a easier computation later 
    """
    h, w = image.shape
    
    paddedImage = np.insert(image, [0, h], False, axis=0)
    paddedImage = np.insert(paddedImage, [0, w], False, axis=1)

    return paddedImage

def genBinMask(image, color):
    """
    Returns a binaryMask in boolean state of the image and the color selected
    """
    binaryMask = np.array(image == color)[:,:,0]
    return binaryMask

def edgeDetection(binaryMask):

    h, w = binaryMask.shape
    
    data = binaryMask.ravel()

    right = np.roll(data, -1)
    left = np.roll(data, 1)
    up = np.roll(data, w)
    down = np.roll(data, -w)

    output = data & (~right | ~left | ~up | ~down)

    return np.reshape(output, (h, w))

color = ColorPalettes.WOODEN[0]
print(color)
binaryMask = genBinMask(IMAGE, color)
binaryMask = padImage(binaryMask)
binaryMask = edgeDetection(binaryMask)

cv.imshow("Mask", np.uint8(255*binaryMask))
cv.waitKey(0)
cv.destroyAllWindows()