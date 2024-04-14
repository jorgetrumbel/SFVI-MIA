import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.uic import loadUi

from PIL import Image as im

import ProgramCommonPaths as PCP

#############################################################
# DialogCaptureView  
class DialogCaptureView(QDialog):
    def __init__(self, parent=None):
        super(DialogCaptureView, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogCaptureView.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        pass

    def setImage(self, image):
        try:
            tempImagePath = PCP.PATH_TEMP_CAPTURE_IMAGE
            data = im.fromarray(image)
            data.save(tempImagePath)
            pixmap = QPixmap(tempImagePath)
            self.captureLabel.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")