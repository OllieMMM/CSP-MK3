# ============================================================================ #
# IMPORTS #

import numpy as np
import cv2 as cv
import time
import ColorPalettes

# ============================================================================ #
# FUNCTIONS #

def buildBrightnessLUT(colorPallet: np.ndarray) -> np.ndarray:
    greyPallet = colorPallet.mean(axis=1)
    grayValues = np.arange(256)[:, None]
    indices = np.argmin(np.abs(grayValues - greyPallet), axis=1)
    return colorPallet[indices]

def brightnessDiffMetric(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return lut[gray]

def colorDiffMetric(image: np.ndarray, lut: np.ndarray, color: str) -> np.ndarray:
    if color == "R":
        channel = 2
    elif color == "G":
        channel = 1
    elif color == "B":
        channel = 0
    else:
        raise ValueError("Invalid color specified")
    channelValues = image[:, :, channel]
    return lut[channelValues]

def buildColorLUT(colorPallet: np.ndarray, color: str) -> np.ndarray:
    if color == "R":
        channel = 2
    elif color == "G":
        channel = 1
    elif color == "B":
        channel = 0
    else:
        raise ValueError("Invalid color specified")
    channelPallet = colorPallet[:, channel]
    channelValues = np.arange(256)[:, None]
    indices = np.argmin(np.abs(channelValues - channelPallet),axis=1)
    return colorPallet[indices]

def buildHSVLUT(colorPallet: np.ndarray) -> np.ndarray:
    hsvPallet = cv.cvtColor(colorPallet[None, :], cv.COLOR_BGR2HSV)

    huePallet = hsvPallet[0, :, 0]
    saturationPallet = hsvPallet[0, :, 1]
    valuePallet = hsvPallet[0, :, 2]

    HSVValues = np.arange(256)[:, None]

    hueIndices = np.argmin(np.abs(HSVValues - huePallet), axis=1)
    saturationIndices = np.argmin(np.abs(HSVValues - saturationPallet), axis=1)
    valueIndices = np.argmin(np.abs(HSVValues - valuePallet), axis=1)

    return colorPallet[hueIndices], colorPallet[saturationIndices], colorPallet[valueIndices]

def hsvDiffMetric(image: np.ndarray, lut: tuple, channel: str) -> np.ndarray:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    if channel == "H":
        return lut[0][hsv[:, :, 0]]
    elif channel == "S":
        return lut[1][hsv[:, :, 1]]
    elif channel == "V":
        return lut[2][hsv[:, :, 2]]
    else:
        raise ValueError("Invalid channel specified")

def saveImage(image, filename):
    cv.imwrite(filename, image)

def smoothFilter(image, lut, k_size, filter="gauss", iterations=1):

    quantized = image

    for _ in range(iterations):
        if filter == "guass":
            quantized = cv.GaussianBlur(quantized, (k_size, k_size), 0)
            quantized = brightnessDiffMetric(quantized, lut)
        if filter == "bilateral":
            quantized = cv.bilateralFilter(quantized, k_size, 75, 75)
            quantized = brightnessDiffMetric(quantized, lut)

    return quantized















# ============================================================================ #
# MAIN #

IMAGE = cv.imread("Image Processing/SheepOG.jpg")

#IMAGE = cv.GaussianBlur(IMAGE, (11, 11), 0)
#IMAGE = cv.medianBlur(IMAGE, 5)
#IMAGE = cv.bilateralFilter(IMAGE, 15, 75, 75)

lut = buildBrightnessLUT(ColorPalettes.WOODEN)

filteredImage = brightnessDiffMetric(IMAGE, lut)

saveImage(filteredImage, "Image Processing/filteredImage.png")

cv.imshow("Quantized", filteredImage)
cv.waitKey(0)


## RESULTS ##
# Average processing time for the 4K twinPM was around 0.07 seconds (15 fps)
# Average processing time for SheepOG was 0.0088 seconds (120 fps)
# The HSV-filtering is kind of bad, as it doesn't seem to seperate the colours well enough
# The brightness and color filtering are much better.
# By builing the LUTs, we can achieve very fast processing times, as the filtering is reduced to simple array indexing.