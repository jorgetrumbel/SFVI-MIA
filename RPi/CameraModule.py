from picamera2 import Picamera2, Preview
from io import BytesIO
from time import sleep
from PIL import Image

#CAMERA SENSOR CONFIGURATIONS
'''
[{'format': SRGGB10_CSI2P, 
'unpacked': 'SRGGB10', 
'bit_depth': 10, 
'size': (1332, 990), 
'fps': 120.05, 
'crop_limits': (696, 528, 2664, 1980), 
'exposure_limits': (31, None)}, 
{'format': SRGGB12_CSI2P, 
'unpacked': 'SRGGB12', 
'bit_depth': 12, 
'size': (2028, 1080), 
'fps': 50.03, 
'crop_limits': (0, 440, 4056, 2160), 
'exposure_limits': (60, 667244877, None)}, 
{'format': SRGGB12_CSI2P, 
'unpacked': 'SRGGB12', 
'bit_depth': 12, 
'size': (2028, 1520), 
'fps': 40.01, 
'crop_limits': (0, 0, 4056, 3040), 
'exposure_limits': (60, 674181621, None)}, 
{'format': SRGGB12_CSI2P, 
'unpacked': 'SRGGB12', 
'bit_depth': 12, 'size': (4056, 3040), 
'fps': 10.0, 
'crop_limits': (0, 0, 4056, 3040), 
'exposure_limits': (114, 674191602, None)}]
'''

#CAMERA CONTROLS
'''
#VALUES AS (MIN,MAX,DEFAULT)
{'ExposureValue': (-8.0, 8.0, 0.0), 
'AeExposureMode': (0, 3, 0), 
'AeMeteringMode': (0, 3, 0), 
'AwbMode': (0, 7, 0), 
'AeFlickerMode': (0, 1, 0), 
'AnalogueGain': (1.0, 22.2608699798584, None), 
'Sharpness': (0.0, 16.0, 1.0), 
'AwbEnable': (False, True, None), 
'Contrast': (0.0, 32.0, 1.0), 
'Saturation': (0.0, 32.0, 1.0), 
'Brightness': (-1.0, 1.0, 0.0), 
'ColourGains': (0.0, 32.0, None), 
'AeFlickerPeriod': (100, 1000000, None), 
'HdrMode': (0, 4, 0), 
'ScalerCrop': ((0, 440, 128, 128), (0, 440, 4056, 2160), (948, 440, 2160, 2160)), 
'ExposureTime': (60, 674181621, None), 
'FrameDurationLimits': (19989, 674193371, None), 
'NoiseReductionMode': (0, 4, 0), 
'AeConstraintMode': (0, 3, 0), 
'AeEnable': (False, True, None)}
'''

class Camera():
    def __init__(self):
        self.stream = BytesIO()
        self.camera = Picamera2()
        self.configureCameraMode()

    def startCamera(self):
        self.camera.start(config = self.config)
        #self.camera.start()
        sleep(1) #CORREGIR

    def stopCamera(self):
        self.camera.stop()
        sleep(1) #CORREGIR

    def takePicture(self):
        self.camera.capture_file(self.stream, format = 'jpeg') #CAMBIAR ESTA FUNCION PARA QUE SEA PARA GUARDAR DIRECTO A UN ARCHIVO
        self.stream.seek(0)
        image = Image.open(self.stream)
        return image

    def takePIL(self):
        image = self.camera.capture_image("main")
        return image

    def takeArray(self):
        imageArray = self.camera.capture_array("main")
        return imageArray

    def configureCameraMode(self):
        mode = self.camera.sensor_modes[1]
        self.config = self.camera.create_still_configuration(buffer_count = 1, sensor = {'output_size': mode['size'], 'bit_depth': mode['bit_depth']})
        self.camera.configure(self.config)

    def setCameraOutputSize(self, height, width):
        self.config["main"]["size"] = (height, width)
        self.camera.configure(self.config)

    def printCameraConfigurations(self):
        #the reported information will require the camera to be stopped
        print(self.camera.sensor_modes)

    def printCameraControls(self):
        print(self.camera.camera_controls)

    def modifyExposureTime(self, exposureTime):
        self.camera.set_controls({"ExposureTime": exposureTime})
        self.config['controls']["ExposureTime"] = exposureTime
        sleep(1)
    
    def modifyAnalogueGain(self, analogueGain):
        self.camera.set_controls({"AnalogueGain": analogueGain})
        self.config['controls']["AnalogueGain"] = analogueGain
        sleep(1)

    def modifyAWB(self, enable, mode):
        self.camera.set_controls({"AwbEnable": enable})
        self.config['controls']["AwbEnable"] = enable
        self.camera.set_controls({"AwbMode": mode})
        self.config['controls']["AwbMode"] = mode
        sleep(1)

    def modifySharpness(self, sharpness):
        self.camera.set_controls({"Sharpness": sharpness})
        self.config['controls']["Sharpness"] = sharpness
        sleep(1)

    def modifyContrast(self, contrast):
        self.camera.set_controls({"Contrast": contrast})
        self.config['controls']["Contrast"] = contrast
        sleep(1)

    def modifySaturation(self, saturation):
        self.camera.set_controls({"Saturation": saturation})
        self.config['controls']["Saturation"] = saturation
        sleep(1)

    def modifyBrightness(self, brightness):
        self.camera.set_controls({"Brightness": brightness})
        self.config['controls']["Brightness"] = brightness
        sleep(1)

    def modifyColourGains(self, colourGains):
        self.camera.set_controls({"ColourGains": colourGains})
        self.config['controls']["ColourGains"] = colourGains
        sleep(1)

    camera = None
    config = None
    stream = None


#TEST CODE
camera = Camera()
camera.setCameraOutputSize(300,300)
camera.startCamera()
#camera.modifyExposureTime(100)
#camera.modifyAnalogueGain(1.0)
#camera.modifyAWB(False, 0)
#camera.modifySharpness(1.0)
#camera.modifyContrast(1.0)
#camera.modifySaturation(1.0)
#camera.modifyBrightness(-0.9)
#camera.modifyColourGains(1.0)

image = camera.takePIL()
image.show()

#camera.modifyExposureTime(5000)
#camera.modifyAnalogueGain(15.0)
#camera.modifyAWB(True, 4)
#camera.modifySharpness(15.0)
#camera.modifyContrast(30.0)
#camera.modifySaturation(30.0)
#camera.modifyBrightness(0.9)
#camera.modifyColourGains(30.0)

#camera.startCamera()
image = camera.takePIL()
image.show()
camera.stopCamera()

#camera.printCameraConfigurations()
#camera.printCameraControls()