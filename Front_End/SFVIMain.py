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
        retVar = 0
        if file:
            self.selectedProgram = file
            self.selectedProgramName = UM.getFileNameFromPath(file)
            if VP.checkIfFileIsVisionProgram(file): #Classic vision program selected
                self.visionProgramStructure.loadProgram(file)
                self.selectedProgramType = VPO.VISION_PROGRAM_TYPES_CLASSIC
                retVar = 1
            elif DLM.checkIfFileIsDLProgram(file):
                self.selectedProgramType = VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING
                self.DLmodel.loadProgram(file)
                retVar = 1
        return retVar


    def programOnlineStatusSet(self, state):
        retVar = 0
        if state in PCO.programOnlineStatusOptions:
            self.programOnlineStatus = state  #Estados permitidos: PCO.programOnlineStatusOptions
            retVar = 1
        return retVar

    def resetProgramCounters(self):
        self.programPicturesTaken = 0
        self.programNOKPictures = 0
        return 1

    def triggerProgramRun(self, file):
        image = 0
        data = 0 
        dataType = 0
        programIndividualResults = 0
        programResult = 0
        self.boardIO.setBusyPinFunc(True)
        if self.selectedProgramType == VPO.VISION_PROGRAM_TYPES_CLASSIC:
            image, data, dataType, programIndividualResults, programResult = self.visionProgramStructure.runProgram(True)
        elif self.selectedProgramType == VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING:
            programResult, image = self.DLmodel.loadedModelPredict(imagePath = file)
        self.programPicturesTaken = self.programPicturesTaken + 1
        if programResult == False:
            self.programNOKPictures = self.programNOKPictures + 1
            self.boardIO.NOKPinFunc()
        else:
            self.boardIO.OKPinFunc()
        self.updateProgramStatusForm()
        self.boardIO.setBusyPinFunc(False)
        return image, data, dataType, programIndividualResults, programResult
    
    def loadVisionProgram(self, programPath):
        retVal = 0
        if checkProgramType(programPath) == PCO.PROGRAM_TYPES_CLASSIC:
            self.visionProgramStructure.loadProgram(programPath)
            retVal = 1
        return retVal
    
    def loadDeepLearningVisionProgram(self, programPath):
        retVal = 0
        if checkProgramType(programPath) == PCO.PROGRAM_TYPES_DEEPLEARNING:
            self.DLmodel.loadProgram(programPath)
            retVal = 1
        return retVal

    def launchNewVisionProgram(self):
        self.visionProgramStructure.clearProgram()
        #CORREGIR
        #self.visionProgramStructure.addInstruction("File Select1", parentItem.text(), VPO.CAPTURE_OPTIONS_FILE_SELECT)
        return 1

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

def getSavedProgramsPathsFromDirectory():
    paths = UM.searchSubfoldersForFilesEndingIn(endString = ".json", path = PCP.PATH_SAVED_PROGRAMS)
    return paths

def checkProgramType(programPath):
    retVal = 0
    if DLM.checkIfFileIsDLProgram(programPath): #DL Program selected
        retVal = PCO.PROGRAM_TYPES_DEEPLEARNING
        #self.DLmodel.loadProgram(programPath)
    elif VP.checkIfFileIsVisionProgram(programPath): #Classic vision program selected
        retVal = PCO.PROGRAM_TYPES_CLASSIC
        #self.visionProgramStructure.loadProgram(programPath)
    return retVal

def copyProgram(file, pathToCopyFile):
    retVal = 0
    try:
        UM.copyProgram(file, pathToCopyFile)
        retVal = 1
    except:
        pass
    return retVal

def deleteProgram(pathToDeleteFile):
    retVal = 0
    try:
        UM.deleteProgram(pathToDeleteFile)
        retVal = 1
    except:
        pass
    return retVal



#ESTO HAY QUE VER COMO SE IMPLEMENTA
'''
def cameraConfigButtonAction(self):
        if UM.isRPi():
            #RPI
            #cameraConfigDialog.loadCameraConfig(self.camera.getControlConfig())
            cameraConfigDialog.loadCameraConfigTransformed(self.camera.getControlConfig())
        else:
            #Windows
            #cameraConfigDialog.loadCameraConfig(CM.getControlDefaults())  
            cameraConfigDialog.loadCameraConfigTransformed(CM.getControlDefaults())  
        cameraConfigDialog.exec()
        cameraOptions = cameraConfigDialog.getFormsValuesTransformed()
        if cameraConfigDialog.getDialogResult():
            #Ok Clicked - Change camera config
            if UM.isRPi():
                self.camera.loadControlConfig(cameraOptions)
'''

#UM.getJsonStringFromFile(selectedItem.text()) #VER SI VALE LA PENA AGREGAR ESTA FUNCION