import sys
import json
import VisionModule as VM
import VisionProgramOptions as VPO
import VisionDetectionModule as VDM

class ProgramStructure():
    def __init__(self):
        pass

    def addInstruction(self, instructionName, parentName, instructionType):
        instructionData = {VPO.INSTRUCTION_DATA_NAME: instructionName,
                            VPO.INSTRUCTION_DATA_TYPE: instructionType,
                            VPO.INSTRUCTION_DATA_PARENT: parentName,
                            VPO.INSTRUCTION_DATA_CONFIGURATION: {}}
        if instructionType in  VPO.captureOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.CAPTURE_CONFIGURATIONS_EXPOSURE: 0}
        elif instructionType in VPO.filterOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS: 0,
                                                VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS: 0,
                                                VPO.FILTER_CONFIGURATIONS_ITERATIONS: 0}
        elif instructionType in VPO.featureDetectionOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1: 50,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2: 50,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3: 50}
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
            if instruction[VPO.INSTRUCTION_DATA_PARENT] == instructionName:
                childrenList.append(instruction[VPO.INSTRUCTION_DATA_NAME])
                for child in self.checkChildren(instruction[VPO.INSTRUCTION_DATA_NAME]):
                    childrenList.append(child)
        #print(childrenList)
        return childrenList

    def checkInstrucionName(self, instructionName):
        return (True == (instructionName in self.programInstructionList))

    def changeInstructionConfiguration(self, instructionName, instructionConfiguration):
        self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION] = instructionConfiguration

    def changeInstructionName(self, instructionID, instructionName):
        pass

    def getInstructionType(self, instructionName):
        return self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_TYPE]
    
    def getInstructionConfiguration(self, instructionName):
        return self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION]

    def getProgram(self):
        return self.programInstructionList

    def saveProgram(self, path):
        with open(path, "w") as write_file:
            json.dump(self.programInstructionList, write_file, indent=4)
        pass

    def runProgram(self):
        #HAY QUE CAMBIAR ESTO, POR AHORA CORRE UN PROGRAMA LINEAL, NO PUEDE TENER BRANCHES (CORRE TODAS LAS INSTRUCCIONES EN SERIE)
        #visionProgram = VisionProgram()
        self.saveProgram("temp/program_file.json")
        image = VM.loadImage("images/sudoku.png", grayscale=True)
        for instruction in self.programInstructionList.values():
            instructionConfiguration = instruction[VPO.INSTRUCTION_DATA_CONFIGURATION]
            if instruction[VPO.INSTRUCTION_DATA_TYPE] == VPO.FILTER_OPTIONS_BLUR:
                image = VM.applyBlurFilter(image,
                                            instructionConfiguration[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS], 
                                            instructionConfiguration[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] == VPO.FILTER_OPTIONS_GAUSS:
                image = VM.applyGaussFilter(image,
                                            instructionConfiguration[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS], 
                                            instructionConfiguration[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] == VPO.FILTER_OPTIONS_SOBEL:
                image = VM.applySobelFilter(image)
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] == VPO.FILTER_OPTIONS_EROSION:
                image = VM.applyErosionOperation(image,
                                                    instructionConfiguration[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS],
                                                    instructionConfiguration[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS],
                                                    instructionConfiguration[VPO.FILTER_CONFIGURATIONS_ITERATIONS])
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] == VPO.FEATURE_DETECTION_OPTIONS_CANNY:
                image = VM.applyCannyEdgeDetection(image,
                                                    instructionConfiguration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1],
                                                    instructionConfiguration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2],
                                                    instructionConfiguration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3])
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] == VPO.FEATURE_DETECTION_OPTIONS_HOUGH:
                image = VDM.applyHoughLineDetection(image,
                                                    instructionConfiguration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1],
                                                    instructionConfiguration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2],
                                                    instructionConfiguration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3])
        return image

    programInstructionList = {}
    program = {}
    programIndex = 0

