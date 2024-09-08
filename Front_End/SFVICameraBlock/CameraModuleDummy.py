try:
    from picamera2 import Picamera2, Preview
except ImportError:
    print("Not working on RPi")

from io import BytesIO
from time import sleep
from PIL import Image
import numpy as np

import SFVICameraBlock.CameraOptions as CO

DEFAULT_IMAGE_PATH = "./images/apple.PNG"

class Camera():
    def __init__(self):
        self.stream = BytesIO()
        try:
            self.camera = Picamera2()
            self.configureCameraMode()
            self.loadControlConfig(getControlDefaults())
        except:
            print("Couldnt create camera object")
        
        
    def startCamera(self):
        try:
            self.camera.start(config = self.config)
            #self.camera.start()
            sleep(1) #CORREGIR
        except:
            print("Error starting camera: Camera already started?")

    def stopCamera(self):
        pass

    def takePicture(self):
        image = Image.open(DEFAULT_IMAGE_PATH)
        return image

    def takePIL(self):
        image = Image.open(DEFAULT_IMAGE_PATH)
        return image

    def takeArray(self):
        image = Image.open(DEFAULT_IMAGE_PATH)
        imageArray = np.asarray(image)
        return imageArray

    def configureCameraMode(self):
        pass

    def setCameraOutputSize(self, height, width):
        pass
    
    def getCameraOutputSize(self):
        return self.config["main"]["size"]

    def printCameraConfigurations(self):
        pass

    def printCameraControls(self):
        pass

    def modifyExposureTime(self, exposureTime):
        pass

    def getExposureTime(self):
        return 0
    
    def modifyAnalogueGain(self, analogueGain):
        pass

    def getAnalogueGain(self):
        return 0.0

    def modifyAWB(self, enable, mode):
        pass

    def getAWBMode(self):
        return 0
    
    def getAWBEnable(self):
        return False

    def modifySharpness(self, sharpness):
        pass

    def getSharpness(self):
        return 0.0

    def modifyContrast(self, contrast):
        pass

    def getContrast(self):
        return 0.0

    def modifySaturation(self, saturation):
        pass

    def getSaturation(self):
        return 0.0

    def modifyBrightness(self, brightness):
        pass

    def getBrightness(self):
        return 0.0

    def modifyColourGains(self, colourGains):
        pass

    def getColourGains(self):
        return 0

    def getControlConfig(self):
        defaultDict = {CO.CAMERA_CONTROL_OUTPUT_HEIGHT_NAME: self.config["main"]["size"][0],
                        CO.CAMERA_CONTROL_OUTPUT_WIDTH_NAME: self.config["main"]["size"][1],
                        CO.CAMERA_CONTROL_EXPOSURE_TIME_NAME: self.config['controls']["ExposureTime"],
                        CO.CAMERA_CONTROL_ANALOG_GAIN_NAME: self.config['controls']["AnalogueGain"],
                        CO.CAMERA_CONTROL_AWB_ENABLE_NAME: self.config['controls']["AwbEnable"],
                        CO.CAMERA_CONTROL_AWB_MODE_NAME: self.config['controls']["AwbMode"],
                        CO.CAMERA_CONTROL_SHARPNESS_NAME: self.config['controls']["Sharpness"],
                        CO.CAMERA_CONTROL_CONTRAST_NAME: self.config['controls']["Contrast"],
                        CO.CAMERA_CONTROL_SATURATION_NAME: self.config['controls']["Saturation"],
                        CO.CAMERA_CONTROL_BRIGHTNESS_NAME: self.config['controls']["Brightness"],}
                        #CO.CAMERA_CONTROL_COLOR_GAIN_NAME: self.config['controls']["ColourGains"]}
        return defaultDict

    def loadControlConfig(self, config):
        pass

    camera = None
    config = None
    stream = None


def getControlDefaults():
    outputHeightDefault = 2028
    outputWidthDefault = 1080
    exposureTimeDefault = 66657
    awbModeDefault = 0
    analogueGainDefault = 8.0
    sharpnessDefault = 1.0
    awbEnableDefault = False
    contrastDefault = 1.0
    saturationDefault = 1.0
    brightnessDefault = 0.0
    colorGainsDefault = 0
    defaultDict = {CO.CAMERA_CONTROL_OUTPUT_HEIGHT_NAME: outputHeightDefault,
                    CO.CAMERA_CONTROL_OUTPUT_WIDTH_NAME: outputWidthDefault,
                    CO.CAMERA_CONTROL_EXPOSURE_TIME_NAME: exposureTimeDefault,
                    CO.CAMERA_CONTROL_ANALOG_GAIN_NAME: analogueGainDefault,
                    CO.CAMERA_CONTROL_AWB_ENABLE_NAME: awbEnableDefault,
                    CO.CAMERA_CONTROL_AWB_MODE_NAME: awbModeDefault,
                    CO.CAMERA_CONTROL_SHARPNESS_NAME: sharpnessDefault,
                    CO.CAMERA_CONTROL_CONTRAST_NAME: contrastDefault,
                    CO.CAMERA_CONTROL_SATURATION_NAME: saturationDefault,
                    CO.CAMERA_CONTROL_BRIGHTNESS_NAME: brightnessDefault,}
                    #CO.CAMERA_CONTROL_COLOR_GAIN_NAME: colorGainsDefault}
    return defaultDict