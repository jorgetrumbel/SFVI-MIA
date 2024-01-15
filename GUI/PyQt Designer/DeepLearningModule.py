import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import shutil

import torch
import torchvision
import torchsummary
import torchmetrics
import ultralytics
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.io import read_image

#Model imports
from ultralytics import YOLO
from torchvision.models import resnet18, ResNet18_Weights

#Local imports
import DeepLearningProgramOptions as DLPO
import DrawModule as DM
import VisionModule as VM

PATH_TEMP_YOLO_V8_DATASET = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\YOLOv8\\datasets\\DLModel"
PATH_TEMP_YOLO_V8_TRAIN = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\YOLOv8\\datasets\\DLModel\\train"
PATH_TEMP_YOLO_V8_TEST = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\YOLOv8\\datasets\\DLModel\\test"
#CORREGIR PATHS PARA QUE NO SEAN GLOBALES

modelOutputClasses = ["OK", "NOK"]
batchSize = 16
classCount = 2
colorChannels = 3
imageWidth = 150
imageHeight = 150

'''
data_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(imageWidth, imageHeight)),
                    torchvision.transforms.ToTensor()
                  ])

train_set = torchvision.datasets.ImageFolder(root = trainPath, transform=data_transforms)
valid_set = torchvision.datasets.ImageFolder(root = testPath, transform=data_transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batchSize, shuffle=True)

#RESNET18
weights = ResNet18_Weights.IMAGENET1K_V1
transforms = weights.transforms()
resnet18_model = resnet18(weights=weights)
if torch.cuda.is_available():
    resnet18_model.to("cuda")
for param in resnet18_model.parameters():
    param.requires_grad = False
last_layer_in_features = resnet18_model.fc.in_features
resnet18_model.fc = torch.nn.Linear(in_features=last_layer_in_features, out_features=2)
'''

class modelDL():
    def __init__(self):
        pass

    def setSelectedModel(self, selectedModel):
        self.modelType = selectedModel

    def setOKPath(self, path):
        self.okPath = path

    def setNOKPath(self, path):
        self.nokPath = path

    def trainModel(self, epochs, trainTestSplit):
        if self.okPath == None or self.nokPath == None:
            print("Error: Image paths have not been set")
            return #return if paths have not been set
        if self.modelType == DLPO.DL_MODEL_NAME_YOLOV8:
            prepareYoloV8Files(self.okPath, self.nokPath, trainTestSplit)
            self.modelPath = trainYoloV8Model(epochs)
        print("Model Training Finished")
    
    def modelPredict(self, imagePath = None):
        predictRet = False
        imageRet = None
        if self.modelPath == None or imagePath == None:
            return predictRet #return if paths have not been set
        if self.modelType == DLPO.DL_MODEL_NAME_YOLOV8:
            predictRet = predictYoloV8Image(modelPath = str(self.modelPath) + "\\weights\\best.pt",imagePath = imagePath)
        imageRet = VM.loadImage(imagePath, grayscale = False)
        imageRet = DM.drawMeasurementResult(imageRet, predictRet)
        return predictRet, imageRet
    
    def getNextAugmentGroupNumber(self):
        keyCounter = 1
        for key in self.augmentGroups.keys():
            if keyCounter not in self.augmentGroups.keys():
                break
            else:
                keyCounter = keyCounter + 1
        return keyCounter
    
    def getAugmentGroupQuantity(self):
        return len(self.augmentGroups.keys())

    def addAugmentGroup(self):
        nextNumber = self.getNextAugmentGroupNumber()
        self.augmentGroups[nextNumber] = {}
        return nextNumber

    def addAugment(self, groupIndex, augmentName):
        augmentConfig = {DLPO.AUGMENT_CONFIG_VARIABLES_1: 0,
                         DLPO.AUGMENT_CONFIG_VARIABLES_2: 0,
                         DLPO.AUGMENT_CONFIG_VARIABLES_3: 0,
                         DLPO.AUGMENT_CONFIG_VARIABLES_4: 0}
        
        self.augmentGroups[groupIndex][augmentName] = augmentConfig

    def removeAugment(self, groupIndex, augmentName:str):
        if augmentName.startswith(DLPO.GROUP_NAME_STRING):
            del self.augmentGroups[groupIndex]
        else:
            del self.augmentGroups[groupIndex][augmentName]

    def changeAugmentConfiguration(self, groupIndex, augmentName, configuration):
        augmentConfig = {DLPO.AUGMENT_CONFIG_VARIABLES_1: configuration[0],
                         DLPO.AUGMENT_CONFIG_VARIABLES_2: configuration[1],
                         DLPO.AUGMENT_CONFIG_VARIABLES_3: configuration[2],
                         DLPO.AUGMENT_CONFIG_VARIABLES_4: configuration[3]}
        self.augmentGroups[groupIndex][augmentName] = augmentConfig

    def getAugmentConfiguration(self, groupIndex, augmentName):
        return (self.augmentGroups[groupIndex][augmentName])
    
    def augmentImages(self, destinationFolder, nRuns):
        transformsList = []
        deleteFolderFiles(destinationFolder + "\\OK")
        deleteFolderFiles(destinationFolder + "\\NOK")
        for group in self.augmentGroups.keys():
            print("Constructing Augment Compose:")
            augmentList = []
            paramList = []
            for augment in self.augmentGroups[group].keys():
                augmentList.append(augment)
                paramList.append(self.augmentGroups[group][augment])
            if len(augmentList) > 0:
                composedAugments = composeAugments(augmentList, paramList)
                print(composedAugments)
                transformsList.append(composedAugments)
        print("Applying Augments")
        #Create destination paths
        try:  
            os.mkdir(destinationFolder + "\\OK")
        except:  
            pass
        try:  
            os.mkdir(destinationFolder + "\\NOK")
        except:  
            pass
        augmentImages(self.okPath, destinationFolder + "\\OK", transformsList, nRuns) #Create ok files
        augmentImages(self.nokPath, destinationFolder + "\\NOK", transformsList, nRuns) #Create nok files
        #Copy original images to destination path
        print("Copying original images")
        copyFolderFiles(self.okPath, destinationFolder + "\\OK")
        copyFolderFiles(self.nokPath, destinationFolder + "\\NOK")
        print("Augment Finished")
        print("Images saved to " + destinationFolder)
    modelType = None
    okPath = None
    nokPath = None
    modelPath = None
    augmentGroups = {}

def trainYoloV8Model(epochs):
    model = YOLO('yolov8n-cls.pt') # load a pretrained model
    result = model.train(data=PATH_TEMP_YOLO_V8_DATASET, epochs = epochs) # Train the model
    print(result)
    return result.save_dir

def prepareYoloV8Files(okPath, nokPath, testTrainSplit):
    if okPath == None or nokPath == None:
        return #return if paths have not been set
    #Delete existing images in train folders
    deletePaths = [PATH_TEMP_YOLO_V8_TRAIN + "\\OK", PATH_TEMP_YOLO_V8_TRAIN + "\\NOK", PATH_TEMP_YOLO_V8_TEST + "\\OK", PATH_TEMP_YOLO_V8_TEST + "\\NOK"]
    for folderPath in deletePaths:
        deleteFolderFiles(folderPath)
    #Split and copy the images
    threshold = testTrainSplit / 100 #Write test Train Split as a float
    for img in os.listdir(okPath): #Split OK Images
        if os.path.isfile(okPath + "\\" + img):
            if random.uniform(0,1) > threshold:
                shutil.copy(okPath + "\\" + img, PATH_TEMP_YOLO_V8_TRAIN + "\\OK")
            else:
                shutil.copy(okPath + "\\" + img, PATH_TEMP_YOLO_V8_TEST + "\\OK")
    for img in os.listdir(nokPath): #Split NOK Images
        if os.path.isfile(nokPath + "\\" + img):
            if random.uniform(0,1) > threshold:
                shutil.copy(nokPath + "\\" + img, PATH_TEMP_YOLO_V8_TRAIN + "\\NOK")
            else:
                shutil.copy(nokPath + "\\" + img, PATH_TEMP_YOLO_V8_TEST + "\\NOK")

def predictYoloV8Image(modelPath, imagePath):
    predictionRet = False
    model = YOLO(modelPath)
    results = model(imagePath)[0]
    resultsP = results.probs
    print(resultsP.data)
    predName = results.names[resultsP.top1]
    print(predName)
    if predName == 'OK':
        predictionRet = True
    return predictionRet

def deleteFolderFiles(folderPath):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except:
            pass

def copyFolderFiles(folderPath, destinationFolder):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path):
                shutil.copy(file_path, destinationFolder)
        except:
            pass

def augmentImages(imgFolder, destinationFolder, transformsList, nRuns):
    fileCounter = 0
    files = [i for i in os.listdir(imgFolder) if i.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    for file in files:
        image = read_image(imgFolder + "\\" + file)
        for nRun in range(nRuns):
            for transforms in transformsList:
                transformedImage = transforms(image)
                fileName = destinationFolder + "\\" + "Augment" + str(fileCounter) + ".png"
                fileCounter = fileCounter + 1
                plt.imsave(fileName,transformedImage.permute(1,2,0).cpu().numpy())
                print(fileName + " created")

def composeAugments(augments, parameters):
    augmentList = []
    for index, augment in enumerate(augments):
        augmentObject = createAugment(augment, parameters[index])
        augmentList.append(augmentObject)
    transforms = v2.Compose(augmentList)
    return transforms

def createAugment(augment, parameters):
    augmentRet = None
    if augment == DLPO.AUGMENT_OPTIONS_RESIZE:
        augmentRet = v2.Resize(size = (parameters[DLPO.AUGMENT_RESIZE_CONFIG_SIZE_H], 
                                       parameters[DLPO.AUGMENT_RESIZE_CONFIG_SIZE_W]))
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_RESIZE:
        augmentRet = v2.RandomResize(min_size = parameters[DLPO.AUGMENT_RANDOM_RESIZE_CONFIG_SIZE_MIN], 
                                     max_size = parameters[DLPO.AUGMENT_RANDOM_RESIZE_CONFIG_SIZE_MAX])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_CROP:
        augmentRet = v2.RandomCrop(size = (parameters[DLPO.AUGMENT_RANDOM_CROP_CONFIG_SIZE_H],
                                           parameters[DLPO.AUGMENT_RANDOM_CROP_CONFIG_SIZE_W]),
                                    padding = parameters[DLPO.AUGMENT_RANDOM_CROP_CONFIG_PADDING])
    elif augment == DLPO.AUGMENT_OPTIONS_CENTER_CROP:
        augmentRet = v2.CenterCrop(size = (parameters[DLPO.AUGMENT_CENTER_CROP_CONFIG_SIZE_H],
                                           parameters[DLPO.AUGMENT_CENTER_CROP_CONFIG_SIZE_W]))
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_HORIZONTAL_FLIP:
        augmentRet = v2.RandomHorizontalFlip(p = parameters[DLPO.AUGMENT_RANDOM_HORIZONTAL_FLIP_CONFIG_PROB] / 100)
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_VERTICAL_FLIP:
        augmentRet = v2.RandomVerticalFlip(p = parameters[DLPO.AUGMENT_RANDOM_VERTICAL_FLIP_CONFIG_PROB] / 100)
    elif augment == DLPO.AUGMENT_OPTIONS_PAD:
        augmentRet = v2.Pad(padding = parameters[DLPO.AUGMENT_PAD_CONFIG_PADDING])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_ROTATION:
        augmentRet = v2.RandomRotation(degrees = parameters[DLPO.AUGMENT_RANDOM_ROTATION_CONFIG_DEGREES])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_AFFINE:
        augmentRet = v2.RandomAffine(degrees = parameters[DLPO.AUGMENT_RANDOM_AFFINE_CONFIG_DEGREES],
                                     translate = parameters[DLPO.AUGMENT_RANDOM_AFFINE_CONFIG_TRANSLATE],
                                     scale = parameters[DLPO.AUGMENT_RANDOM_AFFINE_CONFIG_SCALE],
                                     shear = parameters[DLPO.AUGMENT_RANDOM_AFFINE_CONFIG_SHEAR])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_PERSPECTIVE:
        augmentRet = v2.RandomPerspective(distortion_scale = parameters[DLPO.AUGMENT_RANDOM_PERSPECTIVE_CONFIG_DISTORTION_PERCENT] / 100,
                                          p = parameters[DLPO.AUGMENT_RANDOM_PERSPECTIVE_CONFIG_PROBABILITY])
    elif augment == DLPO.AUGMENT_OPTIONS_COLOR_JITTER:
        augmentRet = v2.ColorJitter(brightness = parameters[DLPO.AUGMENT_COLOR_JITTER_CONFIG_BRIGHTNESS] / 100,
                                    contrast = parameters[DLPO.AUGMENT_COLOR_JITTER_CONFIG_CONTRAST] / 100,
                                    saturation = parameters[DLPO.AUGMENT_COLOR_JITTER_CONFIG_SATURATION] / 100,
                                    hue = parameters[DLPO.AUGMENT_COLOR_JITTER_CONFIG_HUE] / 100)
    elif augment == DLPO.AUGMENT_OPTIONS_GAUSSIAN_BLUR:
        augmentRet = v2.GaussianBlur(kernel_size = parameters[DLPO.AUGMENT_GAUSSIAN_BLUR_CONFIG_KERNEL],
                                     sigma = parameters[DLPO.AUGMENT_GAUSSIAN_BLUR_CONFIG_SIGMA] / 100)
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_INVERT:
        augmentRet = v2.RandomInvert(p = parameters[DLPO.AUGMENT_RANDOM_INVERT_CONFIG_PERCENT] / 100)
    return augmentRet