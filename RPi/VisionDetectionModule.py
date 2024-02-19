import numpy as np
import cv2 as cv
import imutils
import VisionModule as VM
import ImageUtilsModule as IUM
import math
import GeometryModule as GM
from scipy.spatial.distance import cdist

#FEATURE DETECTION FUNCTIONS

def getImageContours(image, minArea = 1, maxArea = 1000000):
    areas = []
    perimeters = []
    centroids = []
    contourBoundingBoxes = []
    minAreaRectangles = []
    ret, thresh = cv.threshold(image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    removeList = []
    for index, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area < minArea or area > maxArea:
                removeList.append(index)
                continue
            perimeter = cv.arcLength(contour,True)
            M = cv.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            contourPoly = cv.approxPolyDP(contour, epsilon = 3, closed = True)
            boundingBox = cv.boundingRect(contourPoly)
            minAreaRectangle = cv.minAreaRect(contour)
            minAreaRectangle = cv.boxPoints(minAreaRectangle)
            areas.append(area)
            perimeters.append(perimeter)
            centroids.append((cx, cy))
            contourBoundingBoxes.append(boundingBox)
            minAreaRectangles.append(minAreaRectangle)
    for index in sorted(removeList, reverse=True):
        del contours[index]
    return [contours, areas, perimeters, centroids, contourBoundingBoxes, minAreaRectangles]
    
def applyHoughLineDetection(image, rho, theta, threshold):
    #imageDraw = cv.Canny(image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
    imageDraw = VM.applyAutoCanny(image)  
    lines = cv.HoughLines(imageDraw, rho = rho, theta = np.pi / 180, threshold = threshold)
    points = []
    indexes = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        points.append([[pt1[0], pt1[1], pt2[0], pt2[1]]])
        
    cleanPoints = GM.cleanOverlappingLines(points, 30)

    index = 0
    for point in points:
        if point in cleanPoints:
            indexes.append(index)
        index = index + 1
    
    lines = lines[indexes]
    distances, angles = calculateLineFeatures(cleanPoints)
    return [lines, cleanPoints, angles]

def applyProbabilisticHoughLineDetection(image, rho, theta, threshold):
    #imageDraw = cv.Canny(image, threshold1 = 50, threshold2 = 200, apertureSize = 3, L2gradient = False)     
    imageDraw = VM.applyAutoCanny(image)  
    lines = cv.HoughLinesP(imageDraw, rho = rho, theta = np.pi / 180, threshold = threshold, minLineLength = 50, maxLineGap = 10)
    lines = GM.cleanOverlappingLines(lines, 10)
    distances, angles = calculateLineFeatures(lines)
    return [lines, distances, angles]

def getLinesWithDetector(image):
    ret, thresh = cv.threshold(image, thresh = 50, maxval = 255, type = cv.THRESH_OTSU)
    lineSegmentDetector = cv.createLineSegmentDetector(refine = cv.LSD_REFINE_ADV, log_eps = 100, density_th = 0.8)
    lines = lineSegmentDetector.detect(thresh)[0]
    distances, angles = calculateLineFeatures(lines)
    return lines, distances, angles

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