import imutils
import cv2 as cv
import numpy as np

def rotateImageWithoutCropping(image, angle):
    image = imutils.rotate_bound(image, angle)
    return image

def scaleImage(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return image

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv.resize(image, (width, height), interpolation = cv.INTER_AREA)
    return result, percent

def getImageRGB(image):
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return img_rgb

def showImage(image):
    cv.imshow("Image", image)
    cv.waitKey(0)

#IMAGE CROPPING UTILITIES
button_down = False
box_points = []

def click_and_crop(event, x, y, flags, param):
    global box_points, button_down
    if (button_down == False) and (event == cv.EVENT_LBUTTONDOWN):
        button_down = True
        box_points = [(x, y)]
    elif (button_down == True) and (event == cv.EVENT_MOUSEMOVE):
        image_copy = param.copy()
        point = (x, y)
        cv.rectangle(image_copy, box_points[0], point, (0, 255, 0), 2)
        cv.imshow("Image Cropper - Press C to Crop", image_copy)
    elif (event == cv.EVENT_LBUTTONUP) and (button_down == True):
        button_down = False
        box_points.append((x, y))
        cv.rectangle(param, box_points[0], box_points[1], (0, 255, 0), 2)
        cv.imshow("Image Cropper - Press C to Crop", param)

def image_crop(image):
    clone = image.copy()
    cv.namedWindow("Image Cropper - Press C to Crop")
    param = image
    cv.setMouseCallback("Image Cropper - Press C to Crop", click_and_crop, param)
    while True:
        cv.imshow("Image Cropper - Press C to Crop", image)
        key = cv.waitKey(1)
        if key == ord("c"):
            cv.destroyAllWindows()
            break
    if len(box_points) == 2:
        cropped_region = clone[box_points[0][1]:box_points[1][1], box_points[0][0]:box_points[1][0]]
    return cropped_region, box_points