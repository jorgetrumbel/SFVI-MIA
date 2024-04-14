import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUi

from PIL import Image as im

import ProgramCommonPaths as PCP
import CameraOptions as CO
import UtilitiesModule as UM

CAPTURE_IMAGE_STRING = "captureImage"

#############################################################
# DialogCameraConfig   
class DialogCameraConfig(QDialog):
    def __init__(self, parent=None):
        super(DialogCameraConfig, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogCameraConfiguration.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        self.buttonSave.clicked.connect(self.saveButtonAction)
        self.ButtonLoad.clicked.connect(self.loadButtonAction)
        self.buttonBox.accepted.connect(self.okButtonAction)
        self.buttonSelectFolder.clicked.connect(self.capturePathName)
        self.buttonCapture.clicked.connect(self.captureButtonAction)

    def updateForms(self):
        self.spinBoxOutputHeight.setValue(self.cameraConfig[CO.CAMERA_CONTROL_OUTPUT_HEIGHT_NAME])
        self.spinBoxOutputWidth.setValue(self.cameraConfig[CO.CAMERA_CONTROL_OUTPUT_WIDTH_NAME])
        self.spinBoxExposureTime.setValue(self.cameraConfig[CO.CAMERA_CONTROL_EXPOSURE_TIME_NAME])
        self.spinBoxAnalogGain.setValue(self.cameraConfig[CO.CAMERA_CONTROL_ANALOG_GAIN_NAME])
        self.spinBoxAWBMode.setValue(self.cameraConfig[CO.CAMERA_CONTROL_AWB_MODE_NAME])
        self.spinBoxSharpness.setValue(self.cameraConfig[CO.CAMERA_CONTROL_SHARPNESS_NAME])
        self.spinBoxContrast.setValue(self.cameraConfig[CO.CAMERA_CONTROL_CONTRAST_NAME])
        self.spinBoxSaturation.setValue(self.cameraConfig[CO.CAMERA_CONTROL_SATURATION_NAME])
        self.spinBoxBrightness.setValue(self.cameraConfig[CO.CAMERA_CONTROL_BRIGHTNESS_NAME])
        #self.spinBoxColorGains.setValue(self.cameraConfig[CO.CAMERA_CONTROL_COLOR_GAIN_NAME])

    def loadCameraConfig(self, cameraDict):
        self.cameraConfig = cameraDict
        self.updateForms()
    
    def loadCameraConfigTransformed(self, cameraDict):
        cameraDict[CO.CAMERA_CONTROL_ANALOG_GAIN_NAME] = int(cameraDict[CO.CAMERA_CONTROL_ANALOG_GAIN_NAME] * 100)
        cameraDict[CO.CAMERA_CONTROL_SHARPNESS_NAME] = int(cameraDict[CO.CAMERA_CONTROL_SHARPNESS_NAME] * 10)
        cameraDict[CO.CAMERA_CONTROL_CONTRAST_NAME] = int(cameraDict[CO.CAMERA_CONTROL_CONTRAST_NAME] * 10)
        cameraDict[CO.CAMERA_CONTROL_SATURATION_NAME] = int(cameraDict[CO.CAMERA_CONTROL_SATURATION_NAME] * 10)
        cameraDict[CO.CAMERA_CONTROL_BRIGHTNESS_NAME] = int(cameraDict[CO.CAMERA_CONTROL_BRIGHTNESS_NAME] * 100)
        #cameraDict[CO.CAMERA_CONTROL_COLOR_GAIN_NAME] = int(cameraDict[CO.CAMERA_CONTROL_COLOR_GAIN_NAME] * 10)               
        self.cameraConfig = cameraDict
        self.updateForms()

    def getFormsValuesTransformed(self):
        cameraDict = {CO.CAMERA_CONTROL_OUTPUT_HEIGHT_NAME: self.spinBoxOutputHeight.value(),
                        CO.CAMERA_CONTROL_OUTPUT_WIDTH_NAME: self.spinBoxOutputWidth.value(),
                        CO.CAMERA_CONTROL_EXPOSURE_TIME_NAME: self.spinBoxExposureTime.value(),
                        CO.CAMERA_CONTROL_ANALOG_GAIN_NAME: self.spinBoxAnalogGain.value() / 100.0,
                        CO.CAMERA_CONTROL_AWB_ENABLE_NAME: self.checkBoxAWBEnable.isChecked(),
                        CO.CAMERA_CONTROL_AWB_MODE_NAME: self.spinBoxAWBMode.value(),
                        CO.CAMERA_CONTROL_SHARPNESS_NAME: self.spinBoxSharpness.value() / 10.0,
                        CO.CAMERA_CONTROL_CONTRAST_NAME: self.spinBoxContrast.value() / 10.0,
                        CO.CAMERA_CONTROL_SATURATION_NAME: self.spinBoxSaturation.value() / 10.0,
                        CO.CAMERA_CONTROL_BRIGHTNESS_NAME: self.spinBoxBrightness.value() / 100.0,}
                        #CO.CAMERA_CONTROL_COLOR_GAIN_NAME: self.spinBoxColorGains.value() / 10.0}
        return cameraDict

    def getFormsValues(self):
        cameraDict = {CO.CAMERA_CONTROL_OUTPUT_HEIGHT_NAME: self.spinBoxOutputHeight.value(),
                        CO.CAMERA_CONTROL_OUTPUT_WIDTH_NAME: self.spinBoxOutputWidth.value(),
                        CO.CAMERA_CONTROL_EXPOSURE_TIME_NAME: self.spinBoxExposureTime.value(),
                        CO.CAMERA_CONTROL_ANALOG_GAIN_NAME: self.spinBoxAnalogGain.value(),
                        CO.CAMERA_CONTROL_AWB_ENABLE_NAME: self.checkBoxAWBEnable.isChecked(),
                        CO.CAMERA_CONTROL_AWB_MODE_NAME: self.spinBoxAWBMode.value(),
                        CO.CAMERA_CONTROL_SHARPNESS_NAME: self.spinBoxSharpness.value(),
                        CO.CAMERA_CONTROL_CONTRAST_NAME: self.spinBoxContrast.value(),
                        CO.CAMERA_CONTROL_SATURATION_NAME: self.spinBoxSaturation.value(),
                        CO.CAMERA_CONTROL_BRIGHTNESS_NAME: self.spinBoxBrightness.value(),}
                        #CO.CAMERA_CONTROL_COLOR_GAIN_NAME: self.spinBoxColorGains.value()}
        return cameraDict

    def saveButtonAction(self):
        config = self.getFormsValues()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, filter = QFileDialog.getSaveFileName(self, caption = "Save File", directory = PCP.PATH_SAVED_CAMERA_CONFIGURATIONS, filter = "Camera Files (*.json);;All Files (*)", options = options)
        if file:
            UM.saveJsonDict(file, config)

    def loadButtonAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, caption = "Select File", directory = PCP.PATH_SAVED_CAMERA_CONFIGURATIONS, filter = "Camera Files (*.json);;All Files (*)", options = options)
        if file:
            self.cameraConfig = UM.loadJsonDict(file)
            self.updateForms()

    def capturePathName(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(self,"Select Directory", options=options)
        if path:
            self.capturePath = path
            files = UM.getDirFiles(path)
            filteredFiles = UM.searchForStringsStartingWith(CAPTURE_IMAGE_STRING, files)
            filteredFiles = UM.removeFileListExtensions(filteredFiles)
            numbers = []
            for file in filteredFiles:
                number = UM.get_trailing_number(file)
                numbers.append(number)
            if len(numbers) > 0:
                self.imageCounter = max(numbers) + 1
            else:
                self.imageCounter = 0
            

    def captureButtonAction(self):
        image = self.captureFunc()
        if image is not None:
            data = im.fromarray(image)
            imagePath = self.capturePath + "/" + CAPTURE_IMAGE_STRING + str(self.imageCounter) + ".png"
            self.imageCounter = self.imageCounter + 1
            data.save(imagePath)

    def getDialogResult(self):
        return self.buttonBoxVal

    def okButtonAction(self):
        self.buttonBoxVal = True

    def setCaptureFunc(self, func):
        self.captureFunc = func

    cameraConfig = {}
    buttonBoxVal = False
    capturePath = None
    captureFunc = None
    imageCounter = 0

# End DialogCommandSelection
#############################################################