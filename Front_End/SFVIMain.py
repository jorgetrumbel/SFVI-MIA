import SFVICommonBlock.UtilitiesModule as UM
import SFVICommonBlock.ProgramCommonPaths as PCP
import SFVICommonBlock.ProgramConfigOptions as PCO

import SFVIClassicVisionBlock.VisionProgramOptions as VPO
import SFVIClassicVisionBlock.VisionProgram as VP
from SFVIClassicVisionBlock.VisionProgram import ProgramStructure

import SFVIDeepLearningBlock.DeepLearningModule as DLM

if UM.isRPi():
    #Working on RPi
    import SFVIGPIOBlock.GPIOModule as IO
    import SFVICameraBlock.CameraModule as CM
else:
    import SFVIGPIOBlock.GPIOModuleDummy as IO
    import SFVICameraBlock.CameraModuleDummy as CM
    import SFVICameraBlock.CameraModulePC as CMPC

class programStatus():

    programOnlineStatus = PCO.PROGRAM_ONLINE_STATUS_OFFLINE
    selectedProgram = None
    selectedProgramName = None
    selectedProgramType = None
    programPicturesTaken = 0
    programNOKPictures = 0
    camera = CM.Camera()
    boardIO = IO.IO()

    #Class attributes
    visionProgramStructure = ProgramStructure()
    DLmodel = DLM.modelDL()

    def resetProgramCounters(self):
        self.programPicturesTaken = 0
        self.programNOKPictures = 0

    def setProgramFileName(self, file):
        if file:
            self.selectedProgram = file
            self.selectedProgramName = UM.getFileNameFromPath(file)
            if VP.checkIfFileIsVisionProgram(file): #Classic vision program selected
                self.visionProgramStructure.loadProgram(file)
                self.selectedProgramType = VPO.VISION_PROGRAM_TYPES_CLASSIC
            elif DLM.checkIfFileIsDLProgram(file):
                self.selectedProgramType = VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING
                self.DLmodel.loadProgram(file)
            print(self.selectedProgramName) #CORREGIR

status = programStatus()


def captureAction():
    image = None
    imageDirectory = PCP.LAST_CAPTURED_IMAGE_PATH

    if UM.isRPi():
        #RPI
        #image = self.camera.takeArray()
        pass #CORREGIR
    else:
        #Windows
        image = CMPC.takePicturePC()
    CMPC.saveCameraImagePC(imageDirectory, image)
    return str(imageDirectory)