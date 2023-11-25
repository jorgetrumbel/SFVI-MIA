import GeometryModule as GM
import numpy as np

#CONTOUR MEASUREMENT FUNCTIONS
def checkContours(contourAreas, contourPerimeters, minArea, maxArea, minPerimeter, maxPerimeter):
    passIndexes = []
    for index in range(1,len(contourAreas)):
        if contourAreas[index] > minArea and contourAreas[index] < maxArea \
        and contourPerimeters[index] > minPerimeter and contourPerimeters[index] < maxPerimeter:
            passIndexes.append(index)
    return passIndexes

def checkPass(indexList):
    retBool = False
    if len(indexList) > 0:
        retBool = True
    return retBool

#LINE MEASUREMENT FUNCTIONS
def checkLineDistance(line1, line2, minDistance, maxDistance):
    lines = []
    distance = 0
    result = False
    point11 = np.array([line1[0][0], line1[0][1]])
    point12 = np.array([line1[0][2], line1[0][3]])
    point21 = np.array([line2[0][0], line2[0][1]])
    point22 = np.array([line2[0][2], line2[0][3]])
    lines.append([point11, point12])
    lines.append([point21, point22])
    distance, points = GM.segmentsMinDistance(lines[0][0][0], lines[0][0][1], lines[0][1][0], lines[0][1][1], lines[1][0][0], lines[1][0][1], lines[1][1][0], lines[1][1][1])
    if distance > minDistance and distance < maxDistance:
        result = True
    return distance, result