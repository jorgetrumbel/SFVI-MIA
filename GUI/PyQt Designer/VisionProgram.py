import sys
import json
import VisionModule as VM
import VisionProgramOptions as VPO
import VisionDetectionModule as VDM
import DrawModule as DM
import GeometryModule as GM
import numpy as np

from PIL import Image as im
import ImageUtilsModule as IUM #FOR DEBUGGING

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
                                                VPO.FILTER_CONFIGURATIONS_ITERATIONS: 0,
                                                VPO.FILTER_CONFIGURATIONS_THRESHOLD: 0,
                                                VPO.FILTER_CONFIGURATIONS_THRESHOLD2: 0}
            
        elif instructionType in VPO.featureDetectionOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1: 0,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2: 0,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3: 0}
        
        elif instructionType in VPO.drawOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_1: 0,
                                                VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_2: 0,
                                                VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_3: 0}
            
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

    def selectTemplate(self):
        image = VM.loadImage("images/hearts_card.png", grayscale=True) #CAMBIAR ESTO
        template = IUM.image_crop(image)
        template = np.array(template)
        tempTemplatePath = "temp/templateCrop.png"
        data = im.fromarray(template)
        data.save(tempTemplatePath)

    def runProgram(self):
        #HAY QUE CAMBIAR ESTO, POR AHORA CORRE UN PROGRAMA LINEAL, NO PUEDE TENER BRANCHES (CORRE TODAS LAS INSTRUCCIONES EN SERIE)
        #visionProgram = VisionProgram()
        self.saveProgram("temp/program_file.json")
        image = VM.loadImage("images/hearts_card.png", grayscale=True)
        for instruction in self.programInstructionList.values():
            instructionConfiguration = instruction[VPO.INSTRUCTION_DATA_CONFIGURATION]
            instructionType = instruction[VPO.INSTRUCTION_DATA_TYPE]
            if instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.filterOptions:
                image = runFilterInstruction(image, instructionType, instructionConfiguration)
            
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.featureDetectionOptions:
                lines, contours, values, locations, templateSize = runFeatureDetectionInstruction(image, instructionType, instructionConfiguration, "temp/templateCrop.png") #CAMBIAR PATH

            elif instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.drawOptions:
                image = runDrawInstruction(image, instructionType, instructionConfiguration, lines, contours, values, locations, templateSize)
        
        return image

    programInstructionList = {}
    program = {}
    programIndex = 0

def runFilterInstruction(image, type, configuration):
    kernelRows = configuration[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS]
    kernelColumns = configuration[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS]
    iterations = configuration[VPO.FILTER_CONFIGURATIONS_ITERATIONS]
    threshold = configuration[VPO.FILTER_CONFIGURATIONS_THRESHOLD]
    threshold2 = configuration[VPO.FILTER_CONFIGURATIONS_THRESHOLD2]
    if type == VPO.FILTER_OPTIONS_BLUR:
        imageRet = VM.applyBlurFilter(image, kernelRows, kernelColumns)
    elif type == VPO.FILTER_OPTIONS_GAUSS:
        imageRet = VM.applyGaussFilter(image, kernelRows, kernelColumns)
    elif type == VPO.FILTER_OPTIONS_SOBEL:
        imageRet = VM.applySobelFilter(image)
    elif type == VPO.FILTER_OPTIONS_MEDIAN:
        imageRet = VM.applyMedianFilter(image, kernelRows)  
    elif type == VPO.FILTER_OPTIONS_EROSION:
        imageRet = VM.applyErosionOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_DILATION:
        imageRet = VM.applyDilationOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_OPEN:
        imageRet = VM.applyOpenOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_CLOSE:
        imageRet = VM.applyCloseOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_GRADIENT:
        imageRet = VM.applyMorphologicalGradientOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_TOPHAT:
        imageRet = VM.applyTopHatOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_BLACKHAT:
        imageRet = VM.applyBlackHatOperation(image, kernelRows, kernelColumns, iterations)
    elif type == VPO.FILTER_OPTIONS_HISTOGRAM:
        imageRet = VM.applyHistogramEqualization(image)  
    elif type == VPO.FILTER_OPTIONS_THRESHOLD:
        imageRet = VM.applyThreshold(image, threshold)
    elif type == VPO.FILTER_OPTIONS_THRESHOLD_RANGE:
        imageRet = VM.applyRangeThreshold(image, threshold, threshold2)
    elif type == VPO.FILTER_OPTIONS_THRESHOLD_OTSU:
        imageRet = VM.applyOtsuThreshold(image)
    elif type == VPO.FILTER_OPTIONS_THRESHOLD_ADAPTATIVE_GAUSSIAN:
        imageRet = VM.applyAdaptativeGaussianThreshold(image, threshold, threshold2)
    elif type == VPO.FILTER_OPTIONS_CANNY:
        imageRet = VM.applyCannyEdgeDetection(image, threshold, threshold2)
    elif type == VPO.FILTER_OPTIONS_CANNY_AUTO:
        imageRet = VM.applyAutoCanny(image)
    return imageRet

def runFeatureDetectionInstruction(image, type, configuration, templatePath):
    contoursRet = None
    linesRet = None
    values = None
    location = None
    var1 = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1]
    var2 = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2]
    var3 = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3]
    template = VM.loadImage(templatePath, grayscale=True)
    templateSize = template.shape
    if type == VPO.FEATURE_DETECTION_OPTIONS_CONTOURS:
        contoursRet = VDM.getImageContours(image)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_HOUGH:
        linesRet = VDM.applyHoughLineDetection(image, var1, var2, var3)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC:
        linesRet = VDM.applyProbabilisticHoughLineDetection(image, var1, var2, var3)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_LINE_DETECTOR:
        linesRet = VDM.getLinesWithDetector(image)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH:
        values, location = VDM.matchTemplate(image, template)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_MULTIPLE:
        values, location = VDM.matchTemplateMultiple(image, template, threshold = var1)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_INVARIANT:
        values, location = VDM.matchTemplateInvariant(image, template, threshold = var1, scaleValues = None, rotationAngles = None) #CORREGIR
    elif type == VPO.FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH:
        values, location = VDM.cannyTemplateMatch(image, template, iterations = var1, threshold = var2)
    elif type == VPO.FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH_INVARIANT:
        values, location = VDM.cannyTemplateMatchInvariant(image, template, iterations = var1, threshold = var2, rotationAngles = None, scaleValues = None)
    return linesRet, contoursRet, values, location, templateSize

def runDrawInstruction(image, type, configuration, lines, contours, values, locations, templateSize):
    imageRet = image
    var1 = configuration[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_1]
    var2 = configuration[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_2]
    var3 = configuration[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_3]
    if type == VPO.DRAW_OPTIONS_BOUNDING_BOXES:
        imageRet = DM.drawBoundingBoxes(image, boundingBoxes = None) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_MIN_AREA_RECTANGLES:
        imageRet = DM.drawMinAreaRectangles(image, minAreaRectangles = None) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_CANNY:
        imageRet = DM.drawCannyOverImage(image, var1, var2)
    elif type == VPO.DRAW_OPTIONS_AUTO_CANNY:
        imageRet = DM.drawAutoCannyOverImage(image)
    elif type == VPO.DRAW_OPTIONS_POINT_DISTANCE:
        imageRet = DM.drawDistance(image, distance = None, pointA = None, pointB = None) #Ver si tiene sentido que se tenga esta funcion
    elif type == VPO.DRAW_OPTIONS_SEGMENT_MIN_DISTANCE:
        imageRet = DM.drawSegmentMinDistance(image, line1 = None, line2 = None)
    elif type == VPO.DRAW_OPTIONS_DETECTED_HOUGH_LINES:
        imageRet = DM.drawDetectedHoughLines(image, lines = lines) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_DETECTED_PROBABILISTIC_HOUGH_LINES:
        cleanLines = GM.cleanOverlappingLines(lines)
        imageRet = DM.drawDetectedProbabilisticHoughLines(image, lines = cleanLines)
    elif type == VPO.DRAW_OPTIONS_SEGMENT_DETECTOR_LINES:
        imageRet = DM.drawSegmentDetectorLines(image, lines = lines) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_TEMPLATE_MATCH:
        imageRet = DM.drawTemplateMatch(image, templateWidth = templateSize[1], templateHeight = templateSize[0], location = locations) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_MULTIPLE_TEMPLATE_MATCH:
        imageRet = DM.drawMultipleTemplateMatch(image, loc = locations, templateWidth = templateSize[1], templateHeight = templateSize[0]) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_TEMPLATE_MATCH_INVARIANT:
        imageRet = DM.drawTemplateMatchInvariant(image, locationList = locations, templateWidth = templateSize[1], templateHeight = templateSize[0]) #CORREGIR
    return imageRet