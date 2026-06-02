import numpy as np
import cv2 as cv

img = cv.imread("Image Processing/hue-saturation-value.webp")
cv.imshow("Original Image", img)

COLOR_PALLET = np.array([[70, 76, 156],[119, 170, 230],[35, 2, 105],[167, 205, 235],[128, 160, 189],[80, 127, 163]], dtype=np.uint8)   
IMAGE = cv.imread("Image Processing/hue-saturation-value.webp")
IMAGE = cv.resize(IMAGE, (0, 0), fx=1, fy=1)
#==============================================================================#
def differnceMetric(image, colorPallet, condition="brightness"):

    height, width, channels = image.shape
    data = image.reshape(height * width, channels) # Flatten the input image to 1D
    data = np.uint8(data)

    if condition == "hue":
        imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        huePallet = cv.cvtColor(colorPallet.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)[:, 0]
        hueImage = imageHSV[:, :, 0].reshape(height * width, 1)
        index_min = closestColor(hueImage, height, width, huePallet)

    elif condition == "saturation":
        imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        saturationPallet = cv.cvtColor(colorPallet.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)[:, 1]
        saturationImage = imageHSV[:, :, 1].reshape(height * width, 1)
        index_min = closestColor(saturationImage, height, width, saturationPallet)

    elif condition == "brightness":
        grayPallet = grayScale(colorPallet)
        grayImage = grayScale(data)
        index_min = closestColor(grayImage, height, width, grayPallet)

    elif condition == "red":
        redPallet = getRed(colorPallet)
        redImage = getRed(data)
        index_min = closestColor(redImage, height, width, redPallet)

    elif condition == "green":
        greenPallet = getGreen(colorPallet)
        greenImage = getGreen(data)
        index_min = closestColor(greenImage, height, width, greenPallet)

    elif condition == "blue":
        bluePallet = getBlue(colorPallet)
        blueImage = getBlue(data)
        index_min = closestColor(blueImage, height, width, bluePallet)
        
    output = colorPallet[index_min]
    output = np.uint8(output.reshape((height, width, channels)))
    return output


def grayScale(flatImage):
    output = np.int16(np.matmul(flatImage, [1/3, 1/3, 1/3])) #Having UINT8 type prevents overflow error
    return output

def getRed(flatImage):
    output = flatImage[:, 0]
    return output

def getGreen(flatImage):
    output = flatImage[:, 1]
    return output

def getBlue(flatImage):
    output = flatImage[:, 2]
    return output

def closestColor(data, height, width, colorPallet):
    data = data.reshape(height * width, 1)
    diff = np.abs(data - colorPallet)
    index_min = np.argmin(diff, axis=1)
    return index_min

################################################################################ 
filteredImage = differnceMetric(IMAGE, COLOR_PALLET, condition="red")
cv.imshow("Filtered Image", filteredImage)
cv.waitKey(0)
cv.destroyAllWindows()