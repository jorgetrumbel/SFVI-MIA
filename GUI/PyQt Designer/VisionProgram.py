import numpy as np
import cv2 as cv
import sys
import json

commandGroups = ("Capture", "Filter")
COMMAND_GROUPS_CAPTURE = commandGroups[0]
COMMAND_GROUPS_FILTER = commandGroups[1]

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

def getImplementedVisionFunctions():
    return filterOptions, captureOptions

class ProgramStructure():
    def __init__(self):
        pass

    def addInstruction(self, instructionName, parentName, instructionType):
        instructionData = {"Name": instructionName,
                            "Type": instructionType,
                            "Parent": parentName,
                            "Configuration": {}}
        if instructionType in  captureOptions:
            instructionData["Configuration"] = {CAPTURE_CONFIGURATIONS_EXPOSURE: 0}
        elif instructionType in filterOptions:
            instructionData["Configuration"] = {FILTER_CONFIGURATIONS_KERNEL_ROWS: 0,
                                                FILTER_CONFIGURATIONS_KERNEL_COLUMNS: 0,
                                                FILTER_CONFIGURATIONS_ITERATIONS: 0}
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
            if instruction['Parent'] == instructionName:
                childrenList.append(instruction['Name'])
                for child in self.checkChildren(instruction['Name']):
                    childrenList.append(child)
        #print(childrenList)
        return childrenList

    def checkInstrucionName(self, instructionName):
        return (True == (instructionName in self.programInstructionList))

    def changeInstructionConfiguration(self, instructionName, instructionConfiguration):
        self.programInstructionList[instructionName]['Configuration'] = instructionConfiguration

    def changeInstructionName(self, instructionID, instructionName):
        pass

    def getInstructionType(self, instructionName):
        return self.programInstructionList[instructionName]['Type']
    
    def getInstructionConfiguration(self, instructionName):
        return self.programInstructionList[instructionName]['Configuration']

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
        visionProgram.loadImage("images/apple.png", grayscale=True)
        for instruction in self.programInstructionList.values():
            instructionConfiguration = instruction["Configuration"]
            if instruction["Type"] == "Blur":
                visionProgram.applyBlurFilter(instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_ROWS], 
                                              instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
            elif instruction["Type"] == "Gauss":
                visionProgram.applyGaussFilter(instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_ROWS], 
                                               instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
            elif instruction["Type"] == "Sobel":
                visionProgram.applySobelFilter()
            elif instruction["Type"] == "Erosion":
                visionProgram.applyErosionOperation(instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_ROWS],
                                                    instructionConfiguration[FILTER_CONFIGURATIONS_KERNEL_COLUMNS],
                                                    instructionConfiguration[FILTER_CONFIGURATIONS_ITERATIONS])
        image = visionProgram.getImage()
        return image

    programInstructionList = {}
    program = {}
    programIndex = 0

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

    def applyMedianFilter(self, ksize):
        self.image = cv.medianBlur(self.image, ksize)

    def applyErosionOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.erode(self.image, kernel = kernel, iterations = iterations)

    def applyDilationOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.dilate(self.image, kernel = kernel, iterations = iterations)

    def applyOpenOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_OPEN, kernel = kernel, iterations = iterations)

    def applyCloseOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_CLOSE, kernel = kernel, iterations = iterations)

    def applyMorphologicalGradientOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_GRADIENT, kernel = kernel, iterations = iterations)

    def applyTopHatOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_TOPHAT, kernel = kernel, iterations = iterations)

    def applyBlackHatOperation(self, kRows = 3, kColumns = 3, iterations = 1):
        kernel = np.ones((kRows, kColumns), np.uint8)
        self.image = cv.morphologyEx(self.image, cv.MORPH_BLACKHAT, kernel = kernel, iterations = iterations)

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