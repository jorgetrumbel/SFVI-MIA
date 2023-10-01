import numpy as np
#import math
import cv2 as cv
import imutils

def loadImage(path, grayscale = False):
    imagePath = path
    if grayscale == False:
        image = cv.imread(imagePath)
    else:
        image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    return image

#BASIC FILTERING OPERATIONS
def applyBlurFilter(image, kRows = 3, kColumns = 3):
    retImage = cv.blur(image, [kRows, kColumns])
    return retImage

def applyGaussFilter(image, kRows = 3, kColumns = 3):
    retImage = cv.GaussianBlur(image, (kRows,kColumns), 5)
    return retImage

def applyMedianFilter(image, ksize):
    retImage = cv.medianBlur(image, ksize)
    return retImage

#ADVANCED FILTERING OPERATIONS
def applySobelFilter(image):
    #FALTA HACER QUE TENGA PARAMETROS, POR AHORA LO HACE AUTOMATICO
    grad_x = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y) 
    retImage = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return retImage

def applyHistogramEqualization(image):
    retImage = cv.equalizeHist(image)
    return retImage

def applyLaplacian(image):
    retImage = cv.Laplacian(image, ddepth = cv.CV_8U)
    return retImage

#MORPHOLOGICAL FILTERING OPERATIONS
def applyErosionOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.erode(image, kernel = kernel, iterations = iterations)
    return retImage

def applyDilationOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.dilate(image, kernel = kernel, iterations = iterations)
    return retImage

def applyOpenOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.morphologyEx(image, cv.MORPH_OPEN, kernel = kernel, iterations = iterations)
    return retImage

def applyCloseOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel = kernel, iterations = iterations)
    return retImage

def applyMorphologicalGradientOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel = kernel, iterations = iterations)
    return retImage

def applyTopHatOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel = kernel, iterations = iterations)

def applyBlackHatOperation(image, kRows = 3, kColumns = 3, iterations = 1):
    kernel = np.ones((kRows, kColumns), np.uint8)
    retImage = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel = kernel, iterations = iterations)
    return retImage
    
#THRESHOLDING OPERATIONS
def applyThreshold(image, threshold):
    retImage = cv.threshold(image, thresh = threshold, maxval = 255, type = cv.THRESH_BINARY)
    return retImage

def applyRangeThreshold(image, lowThresh, highThresh):
    retImage = cv.inRange(image, lowThresh, highThresh)
    return retImage

def applyOtsuThreshold(image):
    ret, retImage = cv.threshold(image , thresh = 0, maxval = 255, type = cv.THRESH_OTSU)
    return retImage

def applyAdaptativeGaussianThreshold(image, kSize = 11, cValue = 2):
    retImage = cv.adaptiveThreshold(image, maxValue = 255, adaptiveMethod = cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv.THRESH_BINARY, blockSize = kSize, C = cValue)
    return retImage

#CANNY FILTERS
def applyCannyEdgeDetection(image, threshold1, threshold2, apertureSize = 3):
    retImage = cv.Canny(image, threshold1 = threshold1, threshold2 = threshold2, apertureSize = 3, L2gradient = False)     
    return retImage

def applyAutoCanny(image):
    retImage = imutils.auto_canny(image)
    return retImage