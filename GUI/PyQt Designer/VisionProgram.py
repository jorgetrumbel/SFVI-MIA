import numpy as np
import cv2 as cv
import sys


class VisionProgram():
    
    def __init__(self):
        pass

    def loadImage(self, path, grayscale = False):
        self.imagePath = path
        if grayscale == False:
            self.image = cv.imread(self.imagePath)
        else:
            self.image = cv.imread(self.imagePath, cv.IMREAD_GRAYSCALE)

    def applyBlurFilter(self, kRows = 3, kColumns = 3):
        self.image = cv.blur(self.image, [kRows, kColumns])

    def applyGaussFilter(self, kRows = 3, kColumns = 3):
        self.image = cv.GaussianBlur(self.image, (kRows,kColumns), 5)

    def applySobelFilter(self):
        grad_x = cv.Sobel(self.image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(self.image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y) 
        self.image = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

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

'''
myProgram = VisionProgram()
myProgram.loadImage("images/apple.png")
myProgram.applyFilter()
myProgram.showImage()
'''