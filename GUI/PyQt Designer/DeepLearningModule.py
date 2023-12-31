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
    
    modelType = None
    okPath = None
    nokPath = None
    modelPath = None

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