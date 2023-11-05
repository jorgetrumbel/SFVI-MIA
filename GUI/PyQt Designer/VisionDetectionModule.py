import numpy as np
import cv2 as cv
import imutils
import VisionModule as VM
import ImageUtilsModule as IUM
import math
import GeometryModule as GM
from scipy.spatial.distance import cdist

#FEATURE DETECTION FUNCTIONS

def getImageContours(image):
    areas = []
    perimeters = []
    centroids = []
    ret, thresh = cv.threshold(image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour,True)
            M = cv.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            areas.append(area)
            perimeters.append(perimeter)
            centroids.append((cx, cy))
    return [contours, areas, perimeters, centroids]
    
def applyHoughLineDetection(image, rho, theta, threshold):
    imageDraw = cv.Canny(image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
    lines = cv.HoughLines(imageDraw, rho = rho, theta = np.pi / 180, threshold = threshold)
    return lines

def applyProbabilisticHoughLineDetection(image, rho, theta, threshold):
    imageDraw = cv.Canny(image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
    lines = cv.HoughLinesP(imageDraw, rho = rho, theta = np.pi / 180, threshold = threshold, minLineLength = 50, maxLineGap = 10)
    lines = GM.cleanOverlappingLines(lines)
    distances, angles = calculateLineFeatures(lines)
    return [lines, distances, angles]

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

#Calculate line features from start and end point
def calculateLineFeatures(lines):
    distances = []
    angles = []
    xUnitVector = [0,0,1,0]
    for line in lines:
        distance = math.dist((line[0][0],line[0][1]),(line[0][2],line[0][3]))
        distances.append(distance)
        angle = GM.angleBetweenLines(line[0], xUnitVector)
        angles.append(angle)
    return distances, angles

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
    threshold = threshold / 100
    result = cv.matchTemplate(image, template,	cv.TM_CCOEFF_NORMED)
    (y, x) = np.where(result >= threshold)
    values, locations = cleanMultipleMatches(result[y,x], (y,x), template.shape)
    return values, locations

def matchTemplateInvariantRotation(image, template, threshold, rotationAngles):
    resultList = []
    resultCoordinateList = []
    threshold = threshold / 100
    for angle in rotationAngles:
        rotatedTemplate = IUM.rotateImageWithoutCropping(template, angle)
        result = cv.matchTemplate(image, rotatedTemplate, cv.TM_CCOEFF_NORMED)
        (y, x) = np.where(result >= threshold)
        resultList.append((result[y,x]))
        resultCoordinateList.append(((y, x)))
    return resultList, resultCoordinateList

def matchTemplateInvariantScale(image, template, threshold, scaleValues):
    resultList = []
    resultCoordinateList = []
    threshold = threshold / 100
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
    threshold = threshold / 100
    for angle in rotationAngles:
        rotatedTemplate = IUM.rotateImageWithoutCropping(template, angle)
        for scale in scaleValues:
            scaledTemplate = IUM.scaleImage(rotatedTemplate, scale)
            result = cv.matchTemplate(image, scaledTemplate, cv.TM_CCOEFF_NORMED)
            (y, x) = np.where(result >= threshold)
            resultList.append((result[y,x]))
            resultCoordinateList.append(((y, x)))
    return resultList, resultCoordinateList

def cannyTemplateMatch(image, template, iterations = 3, threshold = 90):
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
    values, locations = cleanMultipleMatches(values, locations, template.shape) #PROBAR
    return values, locations

def cleanMultipleMatches(values, locations, templateSize):
    retList = []
    locRetX = []
    locRetY = []
    valRet = []
    locList = list(zip(locations[0],locations[1]))
    locList = list(zip(locList,values))
    sortedList = sorted(locList, key=lambda x: (-x[1]))
    retList = []
    retList.append(sortedList[0])
    templateCompareSize = int((templateSize[0]*2)/3)
    for item in retList:
        removeList = []
        for sortedItem in sortedList:
            if cdist(np.array(item[0]).reshape(1,2),np.array(sortedItem[0]).reshape(1,2)) < templateCompareSize: #CORREGIR
                removeList.append(sortedItem)
        for removeItem in removeList:
            sortedList.remove(removeItem)
        if len(sortedList) > 0:
            retList.append(sortedList[0])

    for i in range(0,len(retList)):
        locRetX.append(retList[i][0][0])
        locRetY.append(retList[i][0][1])
        valRet.append(retList[i][1])
    return valRet, [locRetX, locRetY]