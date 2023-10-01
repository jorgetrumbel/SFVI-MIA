import numpy as np
import cv2 as cv
import imutils
import VisionModule as VM
import ImageUtilsModule as IUM

#FEATURE DETECTION FUNCTIONS

def getImageContours(image):
    ret, thresh = cv.threshold(image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(image, contours = contours, contourIdx = -1, color = (0,255,0), thickness = 3)
    return contours
    
def applyHoughLineDetection(image, rho, theta, threshold):
    imageDraw = cv.Canny(image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
    lines = cv.HoughLines(imageDraw, rho = rho, theta = np.pi / 180, threshold = threshold)
    return lines

def applyProbabilisticHoughLineDetection(image, rho, theta, threshold):
    imageDraw = cv.Canny(image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
    lines = cv.HoughLinesP(imageDraw, rho = rho, theta = np.pi / 180, threshold = threshold, minLineLength = 50, maxLineGap = 10)
    return lines

def getLinesWithDetector(image):
    ret, thresh = cv.threshold(image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
    lineSegmentDetector = cv.createLineSegmentDetector(0)
    lines = lineSegmentDetector.detect(thresh)[0]
    return lines

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

# TEMPLATE MATCHING FUNCTIONS

def matchTemplate(image, template):
    #imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #templateGray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    result = cv.matchTemplate(image, template,	cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    return max_val, max_loc

def matchTemplateMultiple(image, template, threshold):
    #imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #templateGray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    result = cv.matchTemplate(image, template,	cv.TM_CCOEFF_NORMED)
    (y, x) = np.where(result >= threshold)
    return result[y,x], (y,x)

def matchTemplateInvariantRotation(image, template, threshold, rotationAngles):
    resultList = []
    resultCoordinateList = []
    for angle in rotationAngles:
        rotatedTemplate = IUM.rotateImageWithoutCropping(template, angle)
        result = cv.matchTemplate(image, rotatedTemplate,	cv.TM_CCOEFF_NORMED)
        (y, x) = np.where(result >= threshold)
        resultList.append((result[y,x]))
        resultCoordinateList.append(((y, x)))
    return resultList, resultCoordinateList

def matchTemplateInvariantScale(image, template, threshold, scaleValues):
    resultList = []
    resultCoordinateList = []
    for scale in scaleValues:
        scaledTemplate = IUM.scaleImage(template, scale)
        result = cv.matchTemplate(image, scaledTemplate, cv.TM_CCOEFF_NORMED)
        (y, x) = np.where(result >= threshold)
        resultList.append((result[y,x]))
        resultCoordinateList.append(((y, x)))
    return resultList, resultCoordinateList

def matchTemplateInvariant(image, template, threshold, scaleValues, rotationAngles):
    resultList = []
    resultCoordinateList = []
    for angle in rotationAngles:
        rotatedTemplate = IUM.rotateImageWithoutCropping(template, angle)
        for scale in scaleValues:
            scaledTemplate = IUM.scaleImage(rotatedTemplate, scale)
            result = cv.matchTemplate(image, scaledTemplate, cv.TM_CCOEFF_NORMED)
            (y, x) = np.where(result >= threshold)
            resultList.append((result[y,x]))
            resultCoordinateList.append(((y, x)))
    return resultList, resultCoordinateList

def cannyTemplateMatch(image, template, iterations = 3, threshold = 0.9):
    image = VM.applyAutoCanny(image)
    image = cv.dilate(image, kernel = (25,25), iterations = iterations)
    template = VM.applyAutoCanny(template)
    template = cv.dilate(template, kernel = (25,25), iterations = iterations)
    values, locations = matchTemplateMultiple(image, template, threshold)
    return values, locations

def cannyTemplateMatchInvariant(image, template, iterations = 3, threshold = 0.9, rotationAngles = [0], scaleValues = [0]):
    image = VM.applyAutoCanny(image)
    image = cv.dilate(image, kernel = (25,25), iterations = iterations)
    template = VM.applyAutoCanny(template)
    template = cv.dilate(template, kernel = (25,25), iterations = iterations)
    values, locations = matchTemplateInvariant(image, template, threshold = threshold, scaleValues = scaleValues, rotationAngles = rotationAngles)
    return values, locations