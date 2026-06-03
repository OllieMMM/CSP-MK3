# ============================================================================ #
# IMPORTS #

import numpy as np
import cv2 as cv
import time

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

def saveImage(image, filename):
    cv.imwrite(filename, image)


IMAGE = cv.imread("Image Processing/SheepOG.jpg")

# ============================================================================ #
# COLOR PALLETS #

WOODEN = np.array([
    [70, 76, 156],
    [119, 170, 230],
    [35, 2, 105],
    [167, 205, 235],
    [128, 160, 189],
    [80, 127, 163]
], dtype=np.uint8)

NATURAL = np.array([
    [ 42,  68,  34],
    [ 78, 132,  76],
    [115, 158, 102],
    [ 55,  82, 126],
    [ 85, 120, 165],
    [180, 145, 105],
    [210, 190, 155],
    [145, 145, 145],
    [195, 195, 195],
    [245, 245, 245],
    [ 30,  26,  22]
], dtype=np.uint8)

SUNSET_DESERT = np.array([
    [77, 94, 255],
    [128, 149, 255],
    [140, 196, 255],
    [79, 116, 216],
    [54, 67, 122],
    [35, 39, 62]
], dtype=np.uint8)

DEEP_OCEAN = np.array([
    [78, 32, 12],
    [126, 72, 24],
    [170, 120, 44],
    [222, 168, 78],
    [255, 216, 142],
    [255, 240, 208]
], dtype=np.uint8)

AUTUMN_FOREST = np.array([
    [11, 95, 58],
    [78, 153, 106],
    [52, 136, 181],
    [36, 102, 205],
    [15, 54, 138],
    [14, 36, 82]
], dtype=np.uint8)

CYBERPUNK_NEON = np.array([
    [30, 20, 20],
    [255, 255, 0],
    [200, 0, 255],
    [255, 0, 180],
    [0, 255, 255],
    [0, 120, 255]
], dtype=np.uint8)

VINTAGE_PASTEL = np.array([
    [189, 214, 234],
    [219, 225, 201],
    [214, 196, 173],
    [181, 182, 216],
    [181, 223, 245],
    [167, 150, 138]
], dtype=np.uint8)

# ============================================================================ #
# MAIN #

start_time = time.time()

lut = buildBrightnessLUT(SUNSET_DESERT)
quantized = brightnessDiffMetric(IMAGE, lut)

end_time = time.time()
print(f"Processing time: {end_time - start_time:.4f} seconds")

saveImage(quantized, "Image Processing/QuantizedGreen.png")

cv.imshow("Quantized", quantized)
cv.waitKey(0)