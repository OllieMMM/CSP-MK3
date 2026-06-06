import numpy as np
import ColorPalettes as cp
import cv2 as cv
import time

class imageVector:

    def __init__ (self, image):
        self.shape = image.shape
        self.channels = self.shape[2]
        self.width = self.shape[1]
        self.height = self.shape[0]
        self.data = image

    def reformat(self):
        '''
        Outputs the image in its proper format
        '''
        return np.reshape(self.data, self.shape)
    
    def greyScale(self):
        '''
        Unreversible compression to (single value) greyscale
        '''
        self.data = cv.cvtColor(self.data, cv.COLOR_BGR2GRAY)
        

    def colorMatch(self, colorPalletLUT):
        self.data = colorPalletLUT[self.data]

    def booleanColorMask(self, color):
        return colorMask(self.data, color)
    

class colorMask:

    def __init__ (self, image, color):
        self.color = color
        self.shape = image.shape
        self.channels = 1
        self.width = self.shape[1]
        self.height = self.shape[0]

        # Generate the mask with an equivalency statement
        self.data = np.array(image == color)[:,:,0] # From BGR [[[True True True]]] take just one boolean

    def visualize(self):
        '''
        Turns all true pixels -> 255 so that the mask can be visulaized easily
        '''
        return np.uint8(255*self.data)
    
    def pad(self):
        self.data = np.insert(self.data, [0, self.height], False, axis=0)
        self.data = np.insert(self.data, [0, self.width], False, axis=1)
        self.width += 2
        self.height += 2


    def edgeDetect(self):
        '''
        Returns the boundaries of the colorMask using a 3x3 kernel
        Process using shifted arrays.
        '''
        self.pad()

        data = self.data.ravel()
        right = np.roll(data, -1)
        left = np.roll(data, 1)
        up = np.roll(data, self.width)
        down = np.roll(data, -self.width)

        output = data & (~right | ~left | ~up | ~down)

        self.data = np.reshape(output, (self.height, self.width))

    def objectStart(self):
        if np.any(self.data):
            index = np.argmax(self.data)
            return [index%self.width, index//self.width]
        else:
            return None
    
    def nextDir(self, prevDir):
        # Placeholder for future direction logic. Return previous direction by default.
        return prevDir
    
    def borderTrace(self):
        """
        Traces out all the borders in a edge filtered binary Image and returns a list of the
        |7   0   1|
        |6   x   2|
        |5   4   3|
        """
        # Initialize
        boundaryList = []
        startPixel = self.objectStart()
        if startPixel == None:
            return None
        nextPoint = startPixel
        borderPixels = np.array([startPixel[0],startPixel[1]], dtype=np.uint16)
        
        neighbourPoints = np.array([[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]]) # Coordinates to 8 by search space around pixel being tested.
        
        DIR_LUT = np.array([
            [6, 7, 0, 1, 2, 3, 4, 5],
            [7, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
            [2, 3, 4, 5, 6, 7, 0, 1],
            [3, 4, 5, 6, 7, 0, 1, 2],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 6, 7, 0, 1, 2, 3, 4]])
        
        searchlist = DIR_LUT[4]

        # Main loop
        while True:
            for i, index in enumerate(searchlist):
                x, y = nextPoint + neighbourPoints[index]
                if self.data[y, x]:
                    borderPixels = np.append(borderPixels, [x, y])
                    nextPoint = [x, y]
                    if nextPoint == startPixel:
                        LENGTH = len(borderPixels)
                        x = borderPixels[np.arange(0, LENGTH, step=2)]
                        y = borderPixels[np.arange(1, LENGTH, step=2)]
                        self.data[y, x] = False
                        
                        if len(borderPixels) >= 100:
                        # Update to handle areas as well as lengths
                            boundaryList.append(borderPixels)    

                        startPixel = self.objectStart()
                        if startPixel == None:
                            return boundaryList
                        borderPixels = np.array([startPixel[0], startPixel[1]], dtype=np.uint16)
                        searchlist = DIR_LUT[4]
                        nextPoint = startPixel
                        break
                    searchlist = DIR_LUT[index]
                    i = 0
                    break
                if i == 7: # Deals with isolated pixels in the image
                    self.data[startPixel[1], startPixel[0]] = False
                    startPixel = self.objectStart()
                    if startPixel == None:
                        return boundaryList
                    nextPoint = startPixel
                    borderPixels = np.array([startPixel[0], startPixel[1]],dtype=np.uint16)
                    i = 0
                    break


def buildBrightnessLUT(colorPallet):
    '''
    Returns the LUT for correct color for all brightness values (0-255).
    '''
    greyPallet = colorPallet.mean(axis=1)
    grayValues = np.arange(256)[:, None]
    indices = np.argmin(np.abs(grayValues - greyPallet), axis=1)
    return colorPallet[indices]
    

def formatPolyline(borderPixels):
    polyline = "<polyline points=\" "
    end = f"\" style=\"fill:None; stroke:red; stroke-width:{0.5}\" />\n"
    for i in range(0, len(borderPixels),2):
        polyline += f"{borderPixels[i]},{borderPixels[i+1]} "
    return polyline + end


def svgFileGenerator(fileName, boundaryList, height, width):
    """
    Generates an SVG file of all of the borders given.
    """
    with open(fileName + ".svg", "w+") as file:
        file.write(f"<svg height=\"{height}\" width=\"{width}\" xmlns=\"http://www.w3.org/2000/svg\">\n")
        for points in boundaryList:
            file.write(formatPolyline(points))
        file.write("</svg>")
    return None

def scaleFactor(target_mm, imageHeight):
    return (target_mm / 25.4) * 96 / imageHeight

# Initialization (Create these objects once)
colors = cp.DEEP_OCEAN
FINAL_HEIGHT = 100 # Final image height measured in mm

IMAGE = cv.imread("Image Processing\TwinPM.jpg")
IMAGE = cv.GaussianBlur(IMAGE, (11,11), 50)
# scale = scaleFactor(87, IMAGE.shape[0])
scale = 1
IMAGE = cv.resize(IMAGE, None, fx=scale, fy=scale)
# Dynamic section, may change, be run multiple times, must be fast!
start = time.time()

IMAGE = imageVector(IMAGE)
LUT = buildBrightnessLUT(colors)
IMAGE.greyScale()
IMAGE.colorMatch(LUT)
MASK0 = IMAGE.booleanColorMask(colors[2])
MASK0.edgeDetect()
boundaryList = MASK0.borderTrace()

svgFileGenerator("OUTPUT", boundaryList, IMAGE.height, IMAGE.width)

process = time.time()

print(f"Time taken: {process - start}s")
cv.imshow("picture", IMAGE.reformat())
cv.waitKey(0)
cv.destroyAllWindows()


# Runs TwinPM.jpg in 2.0sec