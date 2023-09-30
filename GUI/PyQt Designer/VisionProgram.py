import sys
import json
from VisionModule import VisionProgram

instructionDataNames = ("Name", "Type", "Parent", "Configuration")
INSTRUCTION_DATA_NAME = instructionDataNames[0]
INSTRUCTION_DATA_TYPE = instructionDataNames[1]
INSTRUCTION_DATA_PARENT = instructionDataNames[2]
INSTRUCTION_DATA_CONFIGURATION = instructionDataNames[3]

commandGroups = ("Capture", "Filter", "Feature Detection")
COMMAND_GROUPS_CAPTURE = commandGroups[0]
COMMAND_GROUPS_FILTER = commandGroups[1]
COMMAND_GROUPS_FEATURE_DETECTION = commandGroups[2]

filterOptions = ("Blur", "Gauss", "Sobel", "Median", "Erosion", "Dilation", "Open", "Close",
                     "Gradient", "Top Hat", "Black Hat") #GUI names of filters, used for selection purposes
FILTER_OPTIONS_BLUR = filterOptions[0]
FILTER_OPTIONS_GAUSS = filterOptions[1]
FILTER_OPTIONS_SOBEL = filterOptions[2]
FILTER_OPTIONS_MEDIAN = filterOptions[3]
FILTER_OPTIONS_EROSION = filterOptions[4]
FILTER_OPTIONS_DILATION = filterOptions[5]
FILTER_OPTIONS_OPEN = filterOptions[6]
FILTER_OPTIONS_CLOSE = filterOptions[7]
FILTER_OPTIONS_GRADIENT = filterOptions[8]
FILTER_OPTIONS_TOPHAT = filterOptions[9]
FILTER_OPTIONS_BLACKHAT = filterOptions[10]

filterConfigurations = ("Name", "Kernel Rows", "Kernel Columns", "Iterations")
FILTER_CONFIGURATIONS_NAME = filterConfigurations[0]
FILTER_CONFIGURATIONS_KERNEL_ROWS = filterConfigurations[1]
FILTER_CONFIGURATIONS_KERNEL_COLUMNS = filterConfigurations[2]
FILTER_CONFIGURATIONS_ITERATIONS = filterConfigurations[3]

captureOptions = ("Camera", "Flash")
CAPTURE_OPTIONS_CAMERA = captureOptions[0]
CAPTURE_OPTIONS_FLASH = captureOptions[1]

captureConfigurations = ("Name", "Exposure")
CAPTURE_CONFIGURATIONS_NAME = captureConfigurations[0]
CAPTURE_CONFIGURATIONS_EXPOSURE = captureConfigurations[1]

featureDetectionOptions = ("Canny", "Hough")
FEATURE_DETECTION_OPTIONS_CANNY = featureDetectionOptions[0]
FEATURE_DETECTION_OPTIONS_HOUGH = featureDetectionOptions[1]

featureDetectionConfigurations = ("Name", "Variable 1", "Variable 2", "Variable 3")
FEATURE_DETECTION_CONFIGURATIONS_NAME = featureDetectionConfigurations[0]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1 = featureDetectionConfigurations[1]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2 = featureDetectionConfigurations[2]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3 = featureDetectionConfigurations[3]

def getImplementedVisionFunctions():
    return filterOptions, captureOptions

class ProgramStructure():
    def __init__(self):
        pass

    def addInstruction(self, instructionName, parentName, instructionType):
        instructionData = {INSTRUCTION_DATA_NAME: instructionName,
                            INSTRUCTION_DATA_TYPE: instructionType,
                            INSTRUCTION_DATA_PARENT: parentName,
                            INSTRUCTION_DATA_CONFIGURATION: {}}
        if instructionType in  captureOptions:
            instructionData[INSTRUCTION_DATA_CONFIGURATION] = {CAPTURE_CONFIGURATIONS_EXPOSURE: 0}
        elif instructionType in filterOptions:
            instructionData[INSTRUCTION_DATA_CONFIGURATION] = {FILTER_CONFIGURATIONS_KERNEL_ROWS: 0,
                                                FILTER_CONFIGURATIONS_KERNEL_COLUMNS: 0,
                                                FILTER_CONFIGURATIONS_ITERATIONS: 0}
        elif instructionType in featureDetectionOptions:
            instructionData[INSTRUCTION_DATA_CONFIGURATION] = {FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1: 50,
                                                FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2: 50,
                                                FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3: 50}
        self.programInstructionList[instructionName] = instructionData

    def removeInstruction(self, instructionName):
        childrenList = self.checkChildren(instructionName)
        try:
            del self.programInstructionList[instructionName]
            for child in childrenList:
                del self.programInstructionList[child]
        except:
            pass

    def checkChildren(self, instructionName):
        childrenList = []
        for instruction in self.programInstructionList.values():
            if instruction[INSTRUCTION_DATA_PARENT] == instructionName:
                childrenList.append(instruction[INSTRUCTION_DATA_NAME])
                for child in self.checkChildren(instruction[INSTRUCTION_DATA_NAME]):
                    childrenList.append(child)
        #print(childrenList)
        return childrenList

    def checkInstrucionName(self, instructionName):
        return (True == (instructionName in self.programInstructionList))

    def changeInstructionConfiguration(self, instructionName, instructionConfiguration):
        self.programInstructionList[instructionName][INSTRUCTION_DATA_CONFIGURATION] = instructionConfiguration

    def changeInstructionName(self, instructionID, instructionName):
        pass

    def getInstructionType(self, instructionName):
        return self.programInstructionList[instructionName][INSTRUCTION_DATA_TYPE]
    
    def getInstructionConfiguration(self, instructionName):
        return self.programInstructionList[instructionName][INSTRUCTION_DATA_CONFIGURATION]

    def getProgram(self):
        return self.programInstructionList

    def saveProgram(self, path):
        with open(path, "w") as write_file:
            json.dump(self.programInstructionList, write_file, indent=4)
        pass

    def runProgram(self):
        #HAY QUE CAMBIAR ESTO, POR AHORA CORRE UN PROGRAMA LINEAL, NO PUEDE TENER BRANCHES (CORRE TODAS LAS INSTRUCCIONES EN SERIE)
        visionProgram = VisionProgram()
        self.saveProgram("temp/program_file.json")
        visionProgram.loadImage("images/sudoku.png", grayscale=True)
        for instruction in self.programInstructionList.values():
            instructionConfiguration = instruction[INSTRUCTION_DATA_CONFIGURATION]
            if instruction[INSTRUCTION_DATA_TYPE] == FILTER_OPTIONS_BLUR:
                visionProgram.applyBlurFilter(instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_ROWS], 
                                              instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
            elif instruction[INSTRUCTION_DATA_TYPE] == FILTER_OPTIONS_GAUSS:
                visionProgram.applyGaussFilter(instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_ROWS], 
                                               instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
            elif instruction[INSTRUCTION_DATA_TYPE] == FILTER_OPTIONS_SOBEL:
                visionProgram.applySobelFilter()
            elif instruction[INSTRUCTION_DATA_TYPE] == FILTER_OPTIONS_EROSION:
                visionProgram.applyErosionOperation(instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_ROWS],
                                                    instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_COLUMNS],
                                                    instructionConfiguration[FILTER_CONFIGURATIONS_ITERATIONS])
            elif instruction[INSTRUCTION_DATA_TYPE] == FEATURE_DETECTION_OPTIONS_CANNY:
                visionProgram.applyCannyEdgeDetection(instructionConfiguration[FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1],
                                                    instructionConfiguration[FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2],
                                                    instructionConfiguration[FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3])
            elif instruction[INSTRUCTION_DATA_TYPE] == FEATURE_DETECTION_OPTIONS_HOUGH:
                visionProgram.applyHoughLineDetection(instructionConfiguration[FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1],
                                                    instructionConfiguration[FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2],
                                                    instructionConfiguration[FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3])
        image = visionProgram.getImage()
        return image

    programInstructionList = {}
    program = {}
    programIndex = 0

