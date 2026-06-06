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

    def objectStart(self, prevStart):
        start_idx = prevStart[1] * self.width + prevStart[0] + 1

        remaining = self.data.ravel()[start_idx:]

        idx = np.argmax(remaining)

        if not remaining[idx]:
            return None

        idx += start_idx

        return [idx % self.width, idx // self.width]
    
    def contourArea(self, borderPixels):
        pts = np.asarray(borderPixels)

        x = pts[:, 0]
        y = pts[:, 1]

        return 0.5 * abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    
    def borderTrace(self):
        """
        Traces out all the borders in a edge filtered binary Image and returns a list of the directions taken
        |7   0   1|
        |6   x   2|
        |5   4   3|
        """
        # Initialize
        boundaryList = []

        startPixel = self.objectStart([-1, 0])

        if startPixel == None:
            return None
        
        nextPoint_X = startPixel[0]
        nextPoint_Y = startPixel[1]

        borderPixels = [[nextPoint_X,nextPoint_Y],]
        
        # Coordinates to 8 by search space around pixel being tested.
        DX = (-1,-1,0,1,1,1,0,-1)
        DY = (0,-1,-1,-1,0,1,1,1)
        
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
                
                x = nextPoint_X + DX[index]
                y = nextPoint_Y + DY[index]

                if self.data[y, x]:

                    borderPixels.append([x, y])

                    nextPoint_X = x
                    nextPoint_Y = y 

                    if [x, y] == startPixel:

                        coords = np.asarray(borderPixels, dtype=np.int16)
                        self.data[coords[:, 1], coords[:, 0]] = False
                        
                        # Decide which contours to keep! Based on length and area
                        if len(borderPixels) >= 60 and self.contourArea(borderPixels) > 350: # Currently scaled to about 5mm
                            # Min Area = (Max*3.78)^2 pixels^2
                            # Min Length = Max*pi*3.78 pixels
                    
                            boundaryList.append(borderPixels)    

                        startPixel = self.objectStart(startPixel)

                        if startPixel == None:
                            return boundaryList
                        
                        nextPoint_X = startPixel[0]
                        nextPoint_Y = startPixel[1]

                        borderPixels = [[nextPoint_X,nextPoint_Y],]
                        searchlist = DIR_LUT[4]

                        break
                    
                    searchlist = DIR_LUT[index]
                    i = 0
                    break

                if i == 7: # Deals with isolated pixels in the image

                    self.data[startPixel[1], startPixel[0]] = False
                    startPixel = self.objectStart(startPixel)
                    
                    if startPixel == None:
                        return boundaryList
                    
                    nextPoint_X = startPixel[0]
                    nextPoint_Y = startPixel[1]

                    borderPixels = [[nextPoint_X,nextPoint_Y],]
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
    

def formatPolyline(borderPixels, color):
    polyline = "<polyline points=\" "
    end = f"\" style=\"fill:None; stroke:red; stroke-width:{0.5}\" />\n"
    for coord in borderPixels:
        polyline += f"{coord[0]},{coord[1]} "
    return polyline + end


def svgFileGenerator(fileName, boundaryList, height, width):
    """
    Generates an SVG file of all of the borders given.
    """
    with open(fileName + ".svg", "w+") as file:
        file.write(f"<svg height=\"{height}\" width=\"{width}\" xmlns=\"http://www.w3.org/2000/svg\">\n")
        for borderPixels in boundaryList:
            file.write(formatPolyline(borderPixels, colors[3]))
        file.write("</svg>")
    return None

def scaleFactor(target_mm, imageHeight):
    return (target_mm / 25.4) * 96 / imageHeight

# Initialization (Create these objects once)
colors = cp.WOODEN
FINAL_HEIGHT = 100 # Final image height measured in mm
MAX_RES = 5 # 5mm maximum resolution

IMAGE = cv.imread("Image Processing\TwinPM.jpg")
IMAGE = cv.GaussianBlur(IMAGE, (11,11), 50)
scale = scaleFactor(100, IMAGE.shape[0])
# scale = 1
IMAGE = cv.resize(IMAGE, None, fx=scale, fy=scale)
# Dynamic section, may change, be run multiple times, must be fast!
start = time.time()

IMAGE = imageVector(IMAGE)
LUT = buildBrightnessLUT(colors)
IMAGE.greyScale()
IMAGE.colorMatch(LUT)
MASK0 = IMAGE.booleanColorMask(colors[0])
MASK0.edgeDetect()
boundaryList = MASK0.borderTrace()


svgFileGenerator("OUTPUT", boundaryList, IMAGE.height, IMAGE.width)

process = time.time()


print(f"Time taken: {process - start}s")
cv.imshow("picture", IMAGE.reformat())
cv.waitKey(0)
cv.destroyAllWindows()


# RNow uns TwinPM.jpg in 0.5sec 
# Depends on color and scale ofc.