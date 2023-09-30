import numpy as np
#import math
import cv2 as cv
import imutils

class VisionProgram():
    
    def __init__(self):
        pass

    def loadImage(self, path, grayscale = False):
        self.imagePath = path
        if grayscale == False:
            self.image = cv.imread(self.imagePath)
        else:
            self.image = cv.imread(self.imagePath, cv.IMREAD_GRAYSCALE)

    #BASIC FILTERING OPERATIONS
    def applyBlurFilter(self, kRows = 3, kColumns = 3):
        self.image = cv.blur(self.image, [kRows, kColumns])

    def applyGaussFilter(self, kRows = 3, kColumns = 3):
        self.image = cv.GaussianBlur(self.image, (kRows,kColumns), 5)

    def applyMedianFilter(self, ksize):
        self.image = cv.medianBlur(self.image, ksize)

    #ADVANCED FILTERING OPERATIONS
    def applySobelFilter(self):
        #FALTA HACER QUE TENGA PARAMETROS, POR AHORA LO HACE AUTOMATICO
        grad_x = cv.Sobel(self.image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(self.image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y) 
        self.image = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    def applyHistogramEqualization(self):
        self.image = cv.equalizeHist(self.image)

    def applyLaplacian(self):
        self.image = cv.Laplacian(self.image, ddepth = cv.CV_8U)

    #MORPHOLOGICAL FILTERING OPERATIONS
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

    
    #THRESHOLDING OPERATIONS
    def applyThreshold(self, threshold):
        self.image = cv.threshold(self.image, thresh = threshold, maxval = 255, type = cv.THRESH_BINARY)

    def applyRangeThreshold(self, lowThresh, highThresh):
        self.image = cv.inRange(self.image, lowThresh, highThresh)

    def applyOtsuThreshold(self):
        ret, self.image = cv.threshold(self.image , thresh = 0, maxval = 255, type = cv.THRESH_OTSU)

    def applyAdaptativeGaussianThreshold(self, kSize = 11, cValue = 2):
        self.image = cv.adaptiveThreshold(self.image, maxValue = 255, adaptiveMethod = cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv.THRESH_BINARY, blockSize = kSize, C = cValue)

    #AAAAAAAAAAAAAAAAAAAAAAAAAAAA
    def applyCannyEdgeDetection(self, threshold1, threshold2, apertureSize = 3):
        self.image = cv.Canny(self.image, threshold1 = threshold1, threshold2 = threshold2, apertureSize = 3, L2gradient = False)     

    def applyAutoCanny(self):
        self.image = imutils.auto_canny(self.image)

    #FEATURE DETECTION FUNCTIONS

    def getImageContours(self):
        ret, thresh = cv.threshold(self.image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
        self.contours, hierarchy = cv.findContours(thresh, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(image, contours = contours, contourIdx = -1, color = (0,255,0), thickness = 3)
        
    def applyHoughLineDetection(self, rho, theta, threshold):
        image = cv.Canny(self.image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
        self.lines = cv.HoughLines(image, rho = rho, theta = np.pi / 180, threshold = threshold)

    def applyProbabilisticHoughLineDetection(self, rho, theta, threshold):
        image = cv.Canny(self.image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
        self.lines = cv.HoughLinesP(image, rho = rho, theta = np.pi / 180, threshold = threshold, minLineLength = 50, maxLineGap = 10)

    def getLinesWithDetector(self):
        ret, thresh = cv.threshold(self.image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
        lineSegmentDetector = cv.createLineSegmentDetector(0)
        self.lines = lineSegmentDetector.detect(thresh)[0]
        
    #IMAGE MANIPULATION FUNCTIONS

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
    contours = None
    lines = None

#FEATURE CALCULATION FUNCTIONS
def getContoursFeatures(contours, minArea, maxArea):
    contourAreas = []
    contourBoundingBoxes = []
    minAreaRectangles = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < minArea or area > maxArea:
            continue
        contourAreas.append(area)
        #Compute approximation polygon
        contourPoly = cv.approxPolyDP(contour, epsilon = 3, closed = True)
        boundingBox = cv.boundingRect(contourPoly)

        minAreaRectangle = cv.minAreaRect(contour)
        minAreaRectangle = cv.boxPoints(minAreaRectangle)
        #minAreaRectangle = np.array(minAreaRectangle, dtype="int")
        #minAreaRectangle = perspective.order_points(minAreaRectangle)
        '''
        # compute the rotated bounding box of the contour
        box = cv.minAreaRect(contour)
        box = cv.cv.BoxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])
        '''
        #REVISAR PARA AGREGAR OTROS FEATURES QUE DESCRIBAN A LOS CONTOURS
        contourBoundingBoxes.append(boundingBox)
        minAreaRectangles.append(minAreaRectangle)
    return contourAreas, contourBoundingBoxes, minAreaRectangles