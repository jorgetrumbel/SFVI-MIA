import math
import numpy as np

#distance between two segments in the plane:
def segmentsMinDistance(x11, y11, x12, y12, x21, y21, x22, y22):
    if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22): 
      return 0, [[0, 0], [0, 0]]
    # try each of the 4 vertices w/the other segment
    distances = []
    points = []
    distance, point = point_segment_distance(x11, y11, x21, y21, x22, y22)
    distances.append(distance)
    points.append(point)
    distance, point = point_segment_distance(x12, y12, x21, y21, x22, y22)
    distances.append(distance)
    points.append(point)
    distance, point = point_segment_distance(x21, y21, x11, y11, x12, y12)
    distances.append(distance)
    points.append(point)
    distance, point = point_segment_distance(x22, y22, x11, y11, x12, y12)
    distances.append(distance)
    points.append(point)
    index = np.array(distances).argmin()
    return distances[index], points[index]

#Check if two segments in the plane intersect
def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
  dx1 = x12 - x11
  dy1 = y12 - y11
  dx2 = x22 - x21
  dy2 = y22 - y21
  delta = dx2 * dy1 - dy2 * dx1
  if delta == 0: return False  # parallel segments
  s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
  t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
  return (0 <= s <= 1) and (0 <= t <= 1)

#Calculates minimun distance between point and segment
def point_segment_distance(px, py, x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  if dx == dy == 0:  # the segment's just a point
    return math.hypot(px - x1, py - y1), [[px, py], [x1, y1]]

  # Calculate the t that minimizes the distance.
  t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

  # See if this represents one of the segment's end points or a point in the middle.
  if t < 0:
    dx = px - x1
    dy = py - y1
    retX = x1
    retY = y1
  elif t > 1:
    dx = px - x2
    dy = py - y2
    retX = x2
    retY = y2
  else:
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx = px - near_x
    dy = py - near_y
    retX = near_x
    retY = near_y

  return math.hypot(dx, dy), [[px, py], [retX, retY]]


def midpoint(pointA, pointB):
	return (int((pointA[0] + pointB[0]) * 0.5), int((pointA[1] + pointB[1]) * 0.5))

def scaleSegment(line, factor):
  t0 = 0.5 * (1.0 - factor)
  t1 = 0.5 * (1.0 + factor)
  x1 = line[0][0] +(line[0][2] - line[0][0]) * t0
  y1 = line[0][1] +(line[0][3] - line[0][1]) * t0
  x2 = line[0][0] +(line[0][2] - line[0][0]) * t1
  y2 = line[0][1] +(line[0][3] - line[0][1]) * t1
  return [[int(x1), int(y1), int(x2), int(y2)]]

def cleanOverlappingLines(lines):
  retLines = []
  retLines.append(lines[0])
  for line in lines:
    alfa = math.degrees(math.atan2(line[0][2]-line[0][0], line[0][3]-line[0][1]))
    #Scale the line by a factor of 2
    scaledLine = scaleSegment(line, 2)

    if np.all(line == retLines[0]):
      continue

    similar = False
    for c in retLines:
      beta = math.degrees(math.atan2(c[0][2]-c[0][0], c[0][3]-c[0][1]))
      if abs(alfa-beta) <= 2:
        distance1, points = point_segment_distance(c[0][0], c[0][1], scaledLine[0][0], scaledLine[0][1],
                                                    scaledLine[0][2], scaledLine[0][3])
        distance2, points = point_segment_distance(c[0][0], c[0][1], scaledLine[0][0], scaledLine[0][1],
                                                    scaledLine[0][2], scaledLine[0][3])
        if distance1 < 10 and distance2 < 10:
          similar = True
          break
    if not similar:
      retLines.append(line)
  return retLines    