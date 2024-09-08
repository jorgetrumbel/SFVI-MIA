import cv2 as cv

class Camera():
    def __init__(self):
        self.camera = cv.VideoCapture(0)

    def takePicture(self):
        result, image = self.camera.read()
        return result, image

    camera = None

def takePicturePC():
    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv.CAP_PROP_FOURCC, 0x32595559)
    camera.set(cv.CAP_PROP_FPS, 25)
    result, image = camera.read()
    if result:
        camera.release()
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #CORREGIR
        return image
    

def saveCameraImagePC(imageDirectory, image):
    cv.imwrite(imageDirectory, image)
    