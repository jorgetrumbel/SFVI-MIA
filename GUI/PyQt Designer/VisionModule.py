import numpy as np
import cv2 as cv

class VisionProgram():
    
    def __init__(self):
        pass

    def loadImage(self, path, grayscale = False):
        self.imagePath = path
        if grayscale == False:
            self.image = cv.imread(self.imagePath)
        else:
            self.image = cv.imread(self.imagePath, cv.IMREAD_GRAYSCALE)

    def applyBlurFilter(self, kRows = 3, kColumns = 3):
        self.image = cv.blur(self.image, [kRows, kColumns])

    def applyGaussFilter(self, kRows = 3, kColumns = 3):
        self.image = cv.GaussianBlur(self.image, (kRows,kColumns), 5)

    def applyMedianFilter(self, ksize):
        self.image = cv.medianBlur(self.image, ksize)

    def applyErosionOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.erode(self.image, kernel = kernel, iterations = iterations)

    def applyDilationOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.dilate(self.image, kernel = kernel, iterations = iterations)

    def applyOpenOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_OPEN, kernel = kernel, iterations = iterations)

    def applyCloseOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_CLOSE, kernel = kernel, iterations = iterations)

    def applyMorphologicalGradientOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_GRADIENT, kernel = kernel, iterations = iterations)

    def applyTopHatOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_TOPHAT, kernel = kernel, iterations = iterations)

    def applyBlackHatOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_BLACKHAT, kernel = kernel, iterations = iterations)

    def applySobelFilter(self):
        grad_x = cv.Sobel(self.image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(self.image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y) 
        self.image = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    def applyCannyEdgeDetection(self, threshold1, threshold2, apertureSize = 3):
        self.image = cv.Canny(self.image, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize, L2gradient = False)     
    '''
    def applyHoughLineDetection(self, rho, ):
        self.image = cv.HoughLines()
    '''
    def getImageRGB(self):
        img_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        return img_rgb
    
    def getImage(self):
        return self.image

    def showImage(self):
        cv.imshow("Image", self.image)
        cv.waitKey(0)

    image = None
    imagePath = None