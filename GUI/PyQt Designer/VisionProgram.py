import sys
import json
import VisionModule as VM
import VisionProgramOptions as VPO
import VisionDetectionModule as VDM
import DrawModule as DM
import MeasurementModule as MM
import GeometryModule as GM
import CameraModulePC as CM
import numpy as np

from PIL import Image as im
import ImageUtilsModule as IUM #FOR DEBUGGING
import cv2 as cv #FOR DEBUGGING

class ProgramStructure():
    def __init__(self):
        pass

    def addInstruction(self, instructionName, parentName, instructionType):
        instructionData = {VPO.INSTRUCTION_DATA_NAME: instructionName,
                            VPO.INSTRUCTION_DATA_TYPE: instructionType,
                            VPO.INSTRUCTION_DATA_PARENT: parentName,
                            VPO.INSTRUCTION_DATA_IMAGE: None,
                            VPO.INSTRUCTION_DATA_CONFIGURATION: {}}
        
        if instructionType in  VPO.captureOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.FILTER_CONFIGURATIONS_NAME: None,
                                                                   VPO.CAPTURE_CONFIGURATIONS_EXPOSURE: 0,
                                                                   VPO.CAPTURE_CONFIGURATIONS_FILE_PATH: 0}

        elif instructionType in VPO.filterOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {VPO.FILTER_CONFIGURATIONS_NAME: None,
                                                                   VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS: 0,
                                                                   VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS: 0,
                                                                   VPO.FILTER_CONFIGURATIONS_ITERATIONS: 0,
                                                                   VPO.FILTER_CONFIGURATIONS_THRESHOLD: 0,
                                                                   VPO.FILTER_CONFIGURATIONS_THRESHOLD2: 0,
                                                                   VPO.FILTER_CONFIGURATIONS_CROP_AREA: 0}
            
        elif instructionType in VPO.featureDetectionOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {
                                                VPO.FILTER_CONFIGURATIONS_NAME: None,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1: 0,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2: 0,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3: 0,
                                                VPO.FEATURE_DETECTION_CONFIGURATIONS_TEMPLATE_PATH: 0}
        
        elif instructionType in VPO.drawOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {
                                                VPO.FILTER_CONFIGURATIONS_NAME: None,
                                                VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_1: 0,
                                                VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_2: 0,
                                                VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_3: 0}
            
        elif instructionType in VPO.measurementOptions:
            instructionData[VPO.INSTRUCTION_DATA_CONFIGURATION] = {
                                                VPO.FILTER_CONFIGURATIONS_NAME: None,
                                                VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_1: 0,
                                                VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_2: 0,
                                                VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_3: 0,
                                                VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_4: 0}

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
        return childrenList

    def checkInstrucionName(self, instructionName):
        return (True == (instructionName in self.programInstructionList))

    def changeInstructionConfiguration(self, instructionName, instructionConfiguration):
        for key in instructionConfiguration.keys():
            self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION][key] = instructionConfiguration[key]

    def changeInstructionName(self, instructionID, instructionName):
        pass

    def getInstructionType(self, instructionName):
        return self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_TYPE]
    
    def getInstructionConfiguration(self, instructionName):
        return self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION]

    def getProgram(self):
        return self.programInstructionList

    def saveProgram(self, path):
        copyDict = self.programInstructionList
        for key in copyDict.keys():
            try:
                del copyDict[key][VPO.INSTRUCTION_DATA_IMAGE]
            except:
                pass
        with open(path, "w") as write_file:
            json.dump(self.programInstructionList, write_file, indent=4)

    def loadProgram(self, path):
        with open(path, "r") as readFile:
            jsonDict = json.load(readFile)
        self.programInstructionList = jsonDict

    def getProgramAttributes(self):
        parents = []
        instructionNames = []
        instructionConfig = []
        for key in self.programInstructionList.keys():
            parents.append(self.programInstructionList[key][VPO.INSTRUCTION_DATA_PARENT])
            instructionNames.append(self.programInstructionList[key][VPO.INSTRUCTION_DATA_NAME])
            instructionConfig.append(self.programInstructionList[key][VPO.INSTRUCTION_DATA_CONFIGURATION])
        return instructionNames, parents, instructionConfig

    def selectTemplate(self, instructionName, parentInstructionName):
        self.runProgram(False, parentInstructionName)
        image = self.programInstructionList[parentInstructionName][VPO.INSTRUCTION_DATA_IMAGE]
        template, points = IUM.image_crop(image)
        template = np.array(template)
        tempTemplatePath = "temp/" + instructionName + "TemplateCrop.png"
        self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION][VPO.FEATURE_DETECTION_CONFIGURATIONS_TEMPLATE_PATH] = tempTemplatePath
        data = im.fromarray(template)
        data.save(tempTemplatePath)

    def selectCropArea(self, instructionName, parentInstructionName):
        self.runProgram(False, parentInstructionName)
        image = self.programInstructionList[parentInstructionName][VPO.INSTRUCTION_DATA_IMAGE]
        cropImage, points = IUM.image_crop(image)
        cropImage = np.array(cropImage)
        cropImagePath = "temp/" + instructionName + "Crop.png"
        self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION][VPO.FILTER_CONFIGURATIONS_CROP_AREA] = points
        data = im.fromarray(cropImage)
        data.save(cropImagePath)

    def getCaptureFileName(self, instructionName, file):
        self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_CONFIGURATION][VPO.CAPTURE_CONFIGURATIONS_FILE_PATH] = file

    def getAllParentInstructions(self, instructionName):
        retInstructions = {}
        retInstructions[instructionName] = self.programInstructionList[instructionName]
        currentParentName = self.programInstructionList[instructionName][VPO.INSTRUCTION_DATA_PARENT]
        while currentParentName != "":
            retInstructions[currentParentName] = self.programInstructionList[currentParentName]
            currentParentName = self.programInstructionList[currentParentName][VPO.INSTRUCTION_DATA_PARENT]
        retInstructions = dict(reversed(list(retInstructions.items())))
        return retInstructions

    def runProgram(self, runFullProgram, instructionStop = None):
        #HAY QUE REVISAR EL ORDEN DE EJECUCION EN UN PROGRAMA FULL
        instructionsToRun = None
        dataRet = [[0]]
        dataRetType = None
        self.saveProgram("temp/program_file.json")
        if runFullProgram == True:
            instructionsToRun = self.programInstructionList
        else:
            instructionsToRun = self.getAllParentInstructions(instructionStop)
        for instruction in instructionsToRun.values():
            instructionConfiguration = instruction[VPO.INSTRUCTION_DATA_CONFIGURATION]
            instructionType = instruction[VPO.INSTRUCTION_DATA_TYPE]
            instructionParent = instruction[VPO.INSTRUCTION_DATA_PARENT]
            if instructionParent != "":
                image = self.programInstructionList[instructionParent][VPO.INSTRUCTION_DATA_IMAGE].copy()
            if instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.captureOptions:
                #image = VM.loadImage("images/hearts_card.png", grayscale=True) #FOR DEBUGGING
                image = runCaptureInstruction(instructionType, instructionConfiguration)
                instruction[VPO.INSTRUCTION_DATA_IMAGE] = image.copy()

            elif instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.filterOptions:
                image = runFilterInstruction(image, instructionType, instructionConfiguration)
                instruction[VPO.INSTRUCTION_DATA_IMAGE] = image.copy()
            
            elif instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.featureDetectionOptions:
                lines, contours, values, locations, templateSize = runFeatureDetectionInstruction(image, instructionType, instructionConfiguration, "temp/templateCrop.png") #CAMBIAR PATH
                instruction[VPO.INSTRUCTION_DATA_IMAGE] = image.copy()
                dataRet, dataRetType = rearrageResultData(instruction[VPO.INSTRUCTION_DATA_TYPE], lines, contours, values, locations, templateSize)

            elif instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.drawOptions:
                image = runDrawInstruction(image, instructionType, instructionConfiguration, lines, contours, values, locations, templateSize)
                instruction[VPO.INSTRUCTION_DATA_IMAGE] = image.copy()

            elif instruction[VPO.INSTRUCTION_DATA_TYPE] in VPO.measurementOptions:
                image, measurementResult, dataRet, dataRetType = runMeasurementInstruction(image, instructionType, instructionConfiguration, lines, contours, values, locations, templateSize)
                image = DM.drawMeasurementResult(image, measurementResult)
                instruction[VPO.INSTRUCTION_DATA_IMAGE] = image.copy()

        return image, dataRet, dataRetType

    programInstructionList = {}
    program = {}
    programIndex = 0

def runCaptureInstruction(type, configuration):
    exposure = configuration[VPO.CAPTURE_CONFIGURATIONS_EXPOSURE]
    filePath = configuration[VPO.CAPTURE_CONFIGURATIONS_FILE_PATH]
    if type == VPO.CAPTURE_OPTIONS_CAMERA:
        image = CM.takePicturePC()
    elif type == VPO.CAPTURE_OPTIONS_FILE:
        image = VM.loadImage(filePath, grayscale=True)
    elif type == VPO.CAPTURE_OPTIONS_FILE_SELECT:
        filePath = CM.getImageFile()
        image = VM.loadImage(filePath, grayscale=True)
    return image

def runFilterInstruction(image, type, configuration):
    kernelRows = configuration[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS]
    kernelColumns = configuration[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS]
    iterations = configuration[VPO.FILTER_CONFIGURATIONS_ITERATIONS]
    threshold = configuration[VPO.FILTER_CONFIGURATIONS_THRESHOLD]
    threshold2 = configuration[VPO.FILTER_CONFIGURATIONS_THRESHOLD2]
    imageCropArea = configuration[VPO.FILTER_CONFIGURATIONS_CROP_AREA]
    if imageCropArea != 0:
        image = image[imageCropArea[0][1]:imageCropArea[1][1], imageCropArea[0][0]:imageCropArea[1][0]]
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
    contoursRet = [0]
    linesRet = [0]
    values = [0]
    location = [0]
    templateSize = [0]
    var1 = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1]
    var2 = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2]
    var3 = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3]
    templatePath = configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_TEMPLATE_PATH]
    if templatePath != 0:
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
    if type == VPO.DRAW_OPTIONS_CONTOURS:
        imageRet = DM.drawContours(image, contours = contours[0])
        imageRet = DM.drawContoursCentroids(image, points = contours[3])
    elif type == VPO.DRAW_OPTIONS_BOUNDING_BOXES:
        imageRet = DM.drawBoundingBoxes(image, boundingBoxes = contours[4]) 
    elif type == VPO.DRAW_OPTIONS_MIN_AREA_RECTANGLES:
        imageRet = DM.drawMinAreaRectangles(image, minAreaRectangles = contours[5])
    elif type == VPO.DRAW_OPTIONS_CANNY:
        imageRet = DM.drawCannyOverImage(image, var1, var2)
    elif type == VPO.DRAW_OPTIONS_AUTO_CANNY:
        imageRet = DM.drawAutoCannyOverImage(image)
    elif type == VPO.DRAW_OPTIONS_POINT_DISTANCE:
        imageRet = DM.drawDistance(image, distance = None, pointA = None, pointB = None) #Ver si tiene sentido que se tenga esta funcion
    elif type == VPO.DRAW_OPTIONS_SEGMENT_MIN_DISTANCE:
        imageRet = DM.drawSegmentMinDistance(image, line1 = None, line2 = None)
    elif type == VPO.DRAW_OPTIONS_DETECTED_HOUGH_LINES:
        imageRet = DM.drawDetectedHoughLines(image, lines = lines[1])
    elif type == VPO.DRAW_OPTIONS_DETECTED_PROBABILISTIC_HOUGH_LINES:
        imageRet = DM.drawDetectedProbabilisticHoughLines(image, lines = lines[0])
    elif type == VPO.DRAW_OPTIONS_SEGMENT_DETECTOR_LINES:
        imageRet = DM.drawSegmentDetectorLines(image, lines = lines[0])
    elif type == VPO.DRAW_OPTIONS_TEMPLATE_MATCH:
        imageRet = DM.drawTemplateMatch(image, templateWidth = templateSize[1], templateHeight = templateSize[0], location = locations) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_MULTIPLE_TEMPLATE_MATCH:
        imageRet = DM.drawMultipleTemplateMatch(image, loc = locations, templateWidth = templateSize[1], templateHeight = templateSize[0]) #CORREGIR
    elif type == VPO.DRAW_OPTIONS_TEMPLATE_MATCH_INVARIANT:
        imageRet = DM.drawTemplateMatchInvariant(image, locationList = locations, templateWidth = templateSize[1], templateHeight = templateSize[0]) #CORREGIR
    return imageRet

def runMeasurementInstruction(image, type, configuration, lines, contours, values, locations, templateSize):
    imageRet = image
    dataRet = [[0]]
    dataRetType = None
    result = False
    var1 = configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_1]
    var2 = configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_2]
    var3 = configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_3]
    var4 = configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_4]
    if type == VPO.MEASUREMENT_OPTIONS_CONTOURS:
        indeces = MM.checkContours(contourAreas = contours[1], contourPerimeters = contours[2], 
                                   minArea = var1, maxArea = var2, minPerimeter = var3, maxPerimeter = var4)
        contoursArranged, dataType = rearrageResultData(type, lines, contours, values, locations, templateSize)
        if MM.checkPass(indeces):
            result = True
            dataRet = [contoursArranged[i] for i in indeces]
            dataRetType = VPO.FEATURE_DETECTION_OPTIONS_CONTOURS
            imageRet = DM.drawContours(imageRet, contours = [contours[0][i] for i in indeces])
            imageRet = DM.drawContoursCentroids(imageRet, points = [contours[3][i] for i in indeces])
    elif type == VPO.MEASUREMENT_OPTIONS_LINE_DISTANCE:
        lineDistance, result = MM.checkLineDistance(line1 = lines[0][0], line2 = lines[0][1], minDistance = var1, maxDistance = var2)
        if lineDistance > 0:
            imageRet = DM.drawDetectedProbabilisticHoughLines(imageRet, lines[0][0:2])
            imageRet = DM.drawSegmentMinDistance(imageRet, lines[0][0], lines[0][1])
        if result:
            linesArranged, dataType = rearrageResultData(type, lines, contours, values, locations, templateSize)
            dataRet = linesArranged[0:2]
            dataRetType = VPO.FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC
    return imageRet, result, dataRet, dataRetType


def rearrageResultData(instructionType, lines, contours, values, locations, templateSize):
    dataRet = [[0]]
    dataRetType = None
    if instructionType == VPO.FEATURE_DETECTION_OPTIONS_CONTOURS or \
    instructionType == VPO.MEASUREMENT_OPTIONS_CONTOURS:
        dataRet = list(zip(contours[3],contours[2],contours[1]))
        dataRetType = VPO.FEATURE_DETECTION_OPTIONS_CONTOURS
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH:
        dataRet = [[locations,int(values*100)]]
        dataRetType = VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_MULTIPLE or \
    instructionType == VPO.FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH:
        loc = list(zip(locations[0],locations[1]))
        dataRet = list(zip(loc,values))
        dataRetType = VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_HOUGH:                    
        rho = [x[0][0] for x in lines[0]]
        theta = [x[0][1] for x in lines[0]]
        dataRet = list(zip(rho,theta,lines[2]))
        dataRetType = VPO.FEATURE_DETECTION_OPTIONS_HOUGH
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC or \
    instructionType == VPO.MEASUREMENT_OPTIONS_LINE_DISTANCE:                    
        new_list = [x[0] for x in lines[0]]
        startPoints = [(x[0], x[1]) for x in new_list]
        endPoints = [(x[2], x[3]) for x in new_list]
        dataRet = list(zip(startPoints,endPoints,lines[1],lines[2]))
        dataRetType = VPO.FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_LINE_DETECTOR:                    
        new_list = [x[0] for x in lines[0]]
        startPoints = [(int(x[0]), int(x[1])) for x in new_list]
        endPoints = [(int(x[2]), int(x[3])) for x in new_list]
        dataRet = list(zip(startPoints,endPoints,lines[1],lines[2]))
        dataRetType = VPO.FEATURE_DETECTION_OPTIONS_LINE_DETECTOR
    return dataRet, dataRetType

def checkIfFileIsVisionProgram(path):
    retBool = False
    with open(path, "r") as readFile:
        jsonDict = json.load(readFile)
    if VPO.INSTRUCTION_DATA_TYPE in jsonDict[list(jsonDict.keys())[0]].keys():
        retBool = True
    return retBool

def getInstructionVariableNames(instructionType):
    variableNames = []
    if instructionType == VPO.FEATURE_DETECTION_OPTIONS_CONTOURS:
        #variableNames.append(VPO.featureDetectionContoursNames[0])
        pass
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_HOUGH:
        variableNames.append(VPO.featureDetectionHoughLinesNames[0])
        variableNames.append(VPO.featureDetectionHoughLinesNames[1])
        variableNames.append(VPO.featureDetectionHoughLinesNames[2])
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC:
        variableNames.append(VPO.featureDetectionProbabilisticHoughLinesNames[0])
        variableNames.append(VPO.featureDetectionProbabilisticHoughLinesNames[1])
        variableNames.append(VPO.featureDetectionProbabilisticHoughLinesNames[2])
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_LINE_DETECTOR:
        #variableNames.append(VPO.featureDetectionLineDetectorNames[0])
        pass
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH:
        #variableNames.append(VPO.featureDetectionTemplateMatchingNames[0])
        pass
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_MULTIPLE:
        variableNames.append(VPO.featureDetectionTemplateMatchingMultipleNames[0])
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_INVARIANT:
        variableNames.append(VPO.featureDetectionTemplateMatchingInvariantNames[0])
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH:
        variableNames.append(VPO.featureDetectionTemplateMatchingCannyNames[0])
        variableNames.append(VPO.featureDetectionTemplateMatchingCannyNames[1])
    elif instructionType == VPO.FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH_INVARIANT:
        variableNames.append(VPO.featureDetectionTemplateMatchingCannyInvariantNames[0])
        variableNames.append(VPO.featureDetectionTemplateMatchingCannyInvariantNames[1])
        variableNames.append(VPO.featureDetectionTemplateMatchingCannyInvariantNames[2])
    elif instructionType == VPO.DRAW_OPTIONS_CONTOURS:
        #variableNames.append(VPO.drawOptionsContoursNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_BOUNDING_BOXES:
        #variableNames.append(VPO.drawOptionsBoundingBoxesNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_MIN_AREA_RECTANGLES:
        #variableNames.append(VPO.drawOptionsMinAreaRectanglesNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_CANNY:
        variableNames.append(VPO.drawOptionsCannyOverImageNames[0])
        variableNames.append(VPO.drawOptionsCannyOverImageNames[1])
    elif instructionType == VPO.DRAW_OPTIONS_AUTO_CANNY:
        #variableNames.append(VPO.drawOptionsAutoCannyNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_POINT_DISTANCE:
        #variableNames.append(VPO.drawOptionsPointDistanceNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_SEGMENT_MIN_DISTANCE:
        #variableNames.append(VPO.drawOptionsSegmentMinDistanceNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_DETECTED_HOUGH_LINES:
        #variableNames.append(VPO.drawOptionsDetectedHoughLinesNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_DETECTED_PROBABILISTIC_HOUGH_LINES:
        #variableNames.append(VPO.drawOptionsDetectedProbabilisticHoughLinesNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_SEGMENT_DETECTOR_LINES:
        #variableNames.append(VPO.drawOptionsSegmentDetectorLinesNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_TEMPLATE_MATCH:
        #variableNames.append(VPO.drawOptionsTemplateMatchingNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_MULTIPLE_TEMPLATE_MATCH:
        #variableNames.append(VPO.drawOptionsTemplateMatchingMultipleNames[0]) #NOT USED
        pass
    elif instructionType == VPO.DRAW_OPTIONS_TEMPLATE_MATCH_INVARIANT:
        #variableNames.append(VPO.drawOptionsTemplateMatchingInvariantNames[0]) #NOT USED
        pass
    elif instructionType == VPO.MEASUREMENT_OPTIONS_CONTOURS:
        variableNames.append(VPO.measurementOptionsContoursNames[0])
        variableNames.append(VPO.measurementOptionsContoursNames[1])
        variableNames.append(VPO.measurementOptionsContoursNames[2])
        variableNames.append(VPO.measurementOptionsContoursNames[3])
    elif instructionType == VPO.MEASUREMENT_OPTIONS_LINE_DISTANCE:
        variableNames.append(VPO.measurementOptionsLineDistanceNames[0])
        variableNames.append(VPO.measurementOptionsLineDistanceNames[1])
    return variableNames