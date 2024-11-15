#DRAWING FUNCTIONS
import SFVICommonBlock.GeometryModule as GM
import imutils
import cv2 as cv
import numpy as np
import math
from imutils.object_detection import non_max_suppression

def drawContours(image, contours):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    cv.drawContours(image, contours = contours, contourIdx = -1, color = (0,255,0), thickness = 3)
    return image

def drawContoursCentroids(image, points):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    centroidNumber = 0
    for point in points:
        centroidNumber = centroidNumber + 1
        image = cv.drawMarker(image, position = point,  color = (0,255,0), markerType = cv.MARKER_CROSS, markerSize = 10, thickness = 3)
        image = cv.putText(image, text = str(centroidNumber), org = point, fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,255,0), thickness = 2, lineType = cv.LINE_AA)
    return image

def drawContoursInfo(image, points, areas, perimeters):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for point, area, perimeter in points, areas, perimeters:
        text = "Area: " + str(area)
        image = cv.putText(image, text, point - (0,20), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv.LINE_AA)
        text = "Perimeter: " + str(perimeter)
        image = cv.putText(image, text, point - (0,40), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv.LINE_AA)
    return image

def drawBoundingBoxes(image, boundingBoxes):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for box in boundingBoxes:
        cv.rectangle(image, pt1 = (int(box[0]), int(box[1])), pt2 = (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), color = (255, 0, 0), thickness = 2, lineType = cv.FILLED)
        midPoint = GM.midpoint((int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])))
        cv.drawMarker(image, position = (midPoint[0], midPoint[1]), color = (255, 0, 0), markerType = cv.MARKER_CROSS, markerSize = 20, thickness = 2, line_type = cv.FILLED)
    return image

def drawMinAreaRectangles(image, minAreaRectangles):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for box in minAreaRectangles:
        box = np.intp(box)
        cv.drawContours(image, [box], 0, (255, 0, 0))
        midPoint = GM.midpoint(box[1], box[3])
        cv.drawMarker(image, position = (midPoint[0], midPoint[1]), color = (255, 0, 0), markerType = cv.MARKER_CROSS, markerSize = 20, thickness = 2, line_type = cv.FILLED)
    return image

def drawCannyOverImage(image, threshold1, threshold2):
    edges = cv.Canny(image, threshold1 = threshold1, threshold2 = threshold2, apertureSize = 3, L2gradient = False)
    kernel = np.ones((11, 11), np.uint8)
    edges = cv.dilate(edges, kernel = kernel, iterations = 2)
    bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR) #ESTA LINEA HABRIA QUE VER DE PONERLA EN OTRO LADO (SI LA IMAGEN DE ENTRADA NO ES GRAYSCALE TIRA ERROR)
    bgr *= np.array((0,1,0),np.uint8)
    image = np.bitwise_or(image, bgr)
    return image

def drawAutoCannyOverImage(image):
    edges = imutils.auto_canny(image)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.dilate(edges, kernel = kernel, iterations = 2)
    bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    bgr *= np.array((0,1,0),np.uint8)
    image = np.bitwise_or(image, bgr)
    return image

def drawDistance(image, distance, pointA, pointB):
    cv.line(image, (int(pointA[0]), int(pointA[1])) , (int(pointB[0]), int(pointB[1])), (0,255,0), 3, cv.LINE_AA)
    midpointX, midpointY = GM.midpoint(pointA, pointB)
    cv.putText(image, text = "{:.1f}px".format(distance), 
               org = (int(midpointX), int(midpointY - 10)), fontFace = cv.FONT_HERSHEY_SIMPLEX, 
               fontScale = 0.55, color = (0,255,0), thickness = 2)
    return image

def drawSegmentMinDistance(image, line1, line2):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    lines = []
    point11 = np.array([line1[0][0], line1[0][1]])
    point12 = np.array([line1[0][2], line1[0][3]])
    point21 = np.array([line2[0][0], line2[0][1]])
    point22 = np.array([line2[0][2], line2[0][3]])
    lines.append([point11, point12])
    lines.append([point21, point22])
    distance, points = GM.segmentsMinDistance(lines[0][0][0], lines[0][0][1], lines[0][1][0], lines[0][1][1], lines[1][0][0], lines[1][0][1], lines[1][1][0], lines[1][1][1])
    image = drawDistance(image, distance, points[0], points[1])
    return image

def drawDetectedHoughLines(image, lines):
    if len(image.shape) < 3:
        retImage = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for i in range(0, len(lines)):
        l = lines[i][0]
        centerX = int((l[0] + l[2])/2)
        centerY = int((l[1] + l[3])/2)
        cv.line(retImage, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        retImage = cv.drawMarker(retImage, position = (centerX, centerY),  color = (0,255,0), markerType = cv.MARKER_CROSS, markerSize = 10, thickness = 3)
        retImage = cv.putText(retImage, text = str(i+1), org = (centerX, centerY), fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,255,0), thickness = 2, lineType = cv.LINE_AA)
    return retImage

def drawDetectedProbabilisticHoughLines(image, lines):
    if len(image.shape) < 3:
        retImage = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv.line(retImage, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        retImage = cv.drawMarker(retImage, position = (l[0], l[1]),  color = (0,255,0), markerType = cv.MARKER_CROSS, markerSize = 10, thickness = 3)
        retImage = cv.putText(retImage, text = str(i+1), org = (l[0], l[1]), fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,255,0), thickness = 2, lineType = cv.LINE_AA)
    return retImage

def drawSegmentDetectorLines(image, lines):
    lineSegmentDetector = cv.createLineSegmentDetector()
    retImage = lineSegmentDetector.drawSegments(image, lines)
    for i in range(0, len(lines)):
        l = lines[i][0]
        retImage = cv.drawMarker(retImage, position = (int(l[0]), int(l[1])),  color = (0,255,0), markerType = cv.MARKER_CROSS, markerSize = 10, thickness = 3)
        retImage = cv.putText(retImage, text = str(i+1), org = (int(l[0]), int(l[1])), fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,255,0), thickness = 1, lineType = cv.LINE_AA)
    return retImage

def drawMultipleTemplateMatch(image, loc, templateWidth, templateHeight):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    rects = []
    for (x, y) in zip(loc[1], loc[0]):
        rects.append((x, y, x + templateWidth, y + templateHeight))
    #pick = non_max_suppression(np.array(rects), overlapThresh = 0.3)
    pick = rects
    id = 1
    for (startX, startY, endX, endY) in pick:
        image = drawTemplateMatch(image, templateWidth, templateHeight, (startX, startY))
        #Draw center mark and ID text
        centerPoint = (int((startX + endX)/2),int((startY + endY)/2))
        image = cv.drawMarker(image, position = centerPoint,  color = (0,255,0), markerType = cv.MARKER_CROSS, markerSize = 10, thickness = 3)
        image = cv.putText(image, text = str(id), org = centerPoint, fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,255,0), thickness = 2, lineType = cv.LINE_AA)
        id = id + 1
    return image

def drawTemplateMatch(image, templateWidth, templateHeight, location):
    topLeft = location
    bottomRight = topLeft[0] + templateWidth, topLeft[1] + templateHeight
    cv.rectangle(image, pt1 = topLeft, pt2 = bottomRight, color = (0, 255, 0), thickness = 2)
    return image

def drawTemplateMatchInvariant(image, locationList, templateWidth, templateHeight):
    rects = []
    for location in locationList:
        for (x, y) in zip(location[1], location[0]):
            rects.append((x, y, x + templateWidth, y + templateHeight))
    pick = non_max_suppression(np.array(rects), overlapThresh = 0.3)
    for (startX, startY, endX, endY) in pick:
        image = drawTemplateMatch(image, templateWidth, templateHeight, (startX, startY))
    return image

def drawMeasurementResult(image, result):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    top = 5
    bottom = top
    left = 5
    right = left
    if result == True:
        image = cv.copyMakeBorder(image, top, bottom, left, right, borderType = cv.BORDER_CONSTANT, dst = None, value = (0,255,0))
        image = cv.putText(image, text = "PASS", org = (0, 50), fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,255,0), thickness = 2, lineType = cv.LINE_AA)
    else:
        image = cv.copyMakeBorder(image, top, bottom, left, right, borderType = cv.BORDER_CONSTANT, dst = None, value = (0,0,255))
        image = cv.putText(image, text = "FAIL", org = (0, 50), fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv.LINE_AA)
    return image

def drawMeasurementAreaResult(image, result, area):
    if len(image.shape) < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if result == True:
        image = cv.rectangle(image, pt1 = area[0], pt2 = area[1], thickness = 2, color = (0,255,0))
    else:
        image = cv.rectangle(image, pt1 = area[0], pt2 = area[1], thickness = 2, color = (0,0,255))
    return image