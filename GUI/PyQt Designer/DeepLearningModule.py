import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import shutil
import json

import torch
import torchvision
import torchsummary
import torchmetrics
import ultralytics
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.io import read_image
from PIL import Image

#Model imports
from ultralytics import YOLO
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from torchvision.models import RegNet_Y_128GF_Weights, regnet_y_128gf
from torchvision.models import Swin_V2_B_Weights, swin_v2_b

#Local imports
import DeepLearningProgramOptions as DLPO
import DrawModule as DM
import VisionModule as VM

PATH_TEMP_YOLO_V8_DATASET = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\YOLOv8\\datasets\\DLModel"
PATH_TEMP_YOLO_V8_TRAIN = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\YOLOv8\\datasets\\DLModel\\train"
PATH_TEMP_YOLO_V8_TEST = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\YOLOv8\\datasets\\DLModel\\test"
PATH_TEMP_SPLIT_IMAGES = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\splitImages"
PATH_TEMP_SPLIT_IMAGES_TEST = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\splitImages\\test"
PATH_TEMP_SPLIT_IMAGES_TRAIN = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\splitImages\\train"
PATH_TEMP_LOG_TRAIN = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\Log\\Train"
PATH_TEMP_LOG_TEST = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\Log\\Test"
#CORREGIR PATHS PARA QUE NO SEAN GLOBALES (QUE SEAN RELATIVOS)



class modelDL():
    def __init__(self):
        pass

    def setSelectedModel(self, selectedModel):
        self.modelType = selectedModel

    def setOKPath(self, path):
        self.okPath = path

    def setNOKPath(self, path):
        self.nokPath = path

    def getTrainResultGraph(self):
        imageRet = []
        if self.modelResultPath != None:
            imageRet = VM.loadImage(self.modelResultPath, grayscale = False)
        return imageRet

    def trainModel(self, epochs, trainTestSplit, batchSize):
        if self.okPath == None or self.nokPath == None:
            print("Error: Image paths have not been set")
            return #return if paths have not been set
        if self.modelType == DLPO.DL_MODEL_NAME_YOLOV8:
            print("YOLOv8 Model")
            prepareYoloV8Files(self.okPath, self.nokPath, trainTestSplit)
            self.modelPath = trainYoloV8Model(epochs)
        elif self.modelType == DLPO.DL_MODEL_NAME_RESNET18:
            print("RESNET18 Model")
            prepareTorchModelFiles(self.okPath, self.nokPath, trainTestSplit)
            history, self.modelPath, self.modelResultPath = trainResnet18Model(epochs, batchSize, DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_VGG16:
            print("VGG16 Model")
            prepareTorchModelFiles(self.okPath, self.nokPath, trainTestSplit)
            history, self.modelPath, self.modelResultPath = trainVGG16Model(epochs, batchSize, DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_EFFICIENTNET:
            print("EfficientNet Model")
            prepareTorchModelFiles(self.okPath, self.nokPath, trainTestSplit)
            history, self.modelPath, self.modelResultPath = trainEfficientNetModel(epochs, batchSize, DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_REGNET128:
            print("RegNet Model")
            prepareTorchModelFiles(self.okPath, self.nokPath, trainTestSplit)
            history, self.modelPath, self.modelResultPath = trainRegNetModel(epochs, batchSize, DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_SWIN:
            print("Swin Model")
            prepareTorchModelFiles(self.okPath, self.nokPath, trainTestSplit)
            history, self.modelPath, self.modelResultPath = trainSwinModel(epochs, batchSize, DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        print("Model Training Finished")
    
    def modelPredict(self, imagePath = None):
        predictRet = False
        imageRet = None
        if self.modelPath == None or imagePath == None:
            return predictRet #return if paths have not been set
        if self.modelType == DLPO.DL_MODEL_NAME_YOLOV8:
            predictRet = predictYoloV8Image(modelPath = self.modelPath,imagePath = imagePath)
        elif self.modelType == DLPO.DL_MODEL_NAME_RESNET18:
            predictRet = predictResnet18Image(modelPath = self.modelPath, imagePath = imagePath, imageWidth = DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, imageHeight = DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_VGG16:
            predictRet = predictVGG16Image(modelPath = self.modelPath, imagePath = imagePath, imageWidth = DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, imageHeight = DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_EFFICIENTNET:
            predictRet = predictEfficientNetImage(modelPath = self.modelPath, imagePath = imagePath, imageWidth = DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, imageHeight = DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_REGNET128:
            predictRet = predictRegNetImage(modelPath = self.modelPath, imagePath = imagePath, imageWidth = DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, imageHeight = DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        elif self.modelType == DLPO.DL_MODEL_NAME_SWIN:
            predictRet = predictSwinImage(modelPath = self.modelPath, imagePath = imagePath, imageWidth = DLPO.MODEL_TRANSFORM_IMAGE_WIDTH, imageHeight = DLPO.MODEL_TRANSFORM_IMAGE_HEIGHT)
        imageRet = VM.loadImage(imagePath, grayscale = False)
        imageRet = DM.drawMeasurementResult(imageRet, predictRet)
        return predictRet, imageRet
    
    def saveProgram(self, path):
        program = {}
        program[DLPO.DL_PROGRAM_DICT_MODEL] = self.modelType
        program[DLPO.DL_PROGRAM_DICT_OKPATH] = self.okPath
        program[DLPO.DL_PROGRAM_DICT_NOKPATH] = self.nokPath
        program[DLPO.DL_PROGRAM_DICT_MODEL_PATH] = str(self.modelPath)
        program[DLPO.DL_PROGRAM_DICT_MODEL_RESULT_PATH] = str(self.modelResultPath)
        program[DLPO.DL_PROGRAM_DICT_AUGMENT_GROUPS] = self.augmentGroups
        if self.modelPath != None:
            shutil.copy(self.modelPath, str(path + "\\" + os.path.basename(os.path.normpath(self.modelPath))))
            program[DLPO.DL_PROGRAM_DICT_MODEL_PATH] = str(path + "\\" + os.path.basename(os.path.normpath(self.modelPath)))
        if self.modelResultPath != None:
            shutil.copy(self.modelResultPath, path)
            program[DLPO.DL_PROGRAM_DICT_MODEL_RESULT_PATH] = str(path + "\\" + os.path.basename(os.path.normpath(self.modelResultPath)))
        with open(path + "\\DLModelConfig.json", "w") as write_file: #REVISAR - CAMBIAR NOMBRE DE GUARDADO
            json.dump(program, write_file, indent=4)

    def loadProgram(self, path):
        with open(path, "r") as readFile:
            jsonDict = json.load(readFile)
        try:
            self.modelType = jsonDict[DLPO.DL_PROGRAM_DICT_MODEL]
            self.okPath = jsonDict[DLPO.DL_PROGRAM_DICT_OKPATH]
            self.nokPath = jsonDict[DLPO.DL_PROGRAM_DICT_NOKPATH]
            self.modelPath = jsonDict[DLPO.DL_PROGRAM_DICT_MODEL_PATH]
            self.modelResultPath = jsonDict[DLPO.DL_PROGRAM_DICT_MODEL_RESULT_PATH]
            self.augmentGroups = jsonDict[DLPO.DL_PROGRAM_DICT_AUGMENT_GROUPS]
        except:
            print("Load Program error")

    def getProgramAttributes(self):
        groups = []
        augments = []
        augmentsConfig = []
        for group in self.augmentGroups.keys():
            groupAugments = self.augmentGroups[group]
            for augment in groupAugments.keys():
                augments.append(augment)
                groups.append(group)
                augmentsConfig.append(groupAugments[augment])
        return groups, augments, augmentsConfig


    def getNextAugmentGroupNumber(self):
        keyCounter = 1
        for key in self.augmentGroups.keys():
            if str(keyCounter) not in self.augmentGroups.keys():
                break
            else:
                keyCounter = keyCounter + 1
        return str(keyCounter)
    
    def getAugmentGroupQuantity(self):
        return len(self.augmentGroups.keys())

    def addAugmentGroup(self):
        nextNumber = self.getNextAugmentGroupNumber()
        self.augmentGroups[nextNumber] = {}
        return nextNumber

    def addAugment(self, groupIndex:str, augmentName):
        augmentConfig = {DLPO.AUGMENT_CONFIG_VARIABLES_1_KEY: 0,
                         DLPO.AUGMENT_CONFIG_VARIABLES_2_KEY: 0,
                         DLPO.AUGMENT_CONFIG_VARIABLES_3_KEY: 0,
                         DLPO.AUGMENT_CONFIG_VARIABLES_4_KEY: 0}
        
        self.augmentGroups[groupIndex][augmentName] = augmentConfig

    def removeAugment(self, groupIndex:str, augmentName:str):
        if augmentName.startswith(DLPO.GROUP_NAME_STRING):
            del self.augmentGroups[groupIndex]
        else:
            del self.augmentGroups[groupIndex][augmentName]

    def changeAugmentConfiguration(self, groupIndex:str, augmentName:str, configuration):
        augmentConfig = {DLPO.AUGMENT_CONFIG_VARIABLES_1_KEY: configuration[0],
                         DLPO.AUGMENT_CONFIG_VARIABLES_2_KEY: configuration[1],
                         DLPO.AUGMENT_CONFIG_VARIABLES_3_KEY: configuration[2],
                         DLPO.AUGMENT_CONFIG_VARIABLES_4_KEY: configuration[3]}
        self.augmentGroups[groupIndex][augmentName] = augmentConfig

    def getAugmentConfiguration(self, groupIndex:str, augmentName):
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
    modelResultPath = None
    augmentGroups = {}

#########################################################
#YOLOV8
def trainYoloV8Model(epochs):
    model = YOLO('yolov8n-cls.pt') # load a pretrained model
    result = model.train(data=PATH_TEMP_YOLO_V8_DATASET, epochs = epochs) # Train the model
    print(result)
    return str(result.save_dir) + "\\weights\\best.pt"

def prepareYoloV8Files(okPath, nokPath, testTrainSplit):
    if okPath == None or nokPath == None:
        print("Error: Image paths not found")
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
#YOLOV8 END
#########################################################

#########################################################
#RESNET18
def createResnet18StandardModel():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    if torch.cuda.is_available():
        model.to("cuda")

    for param in model.parameters():
        param.requires_grad = False
    last_layer_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=last_layer_in_features, out_features = DLPO.CLASS_QUANTITY)
    return model

def trainResnet18Model(epochs, batchSize, imageWidth, imageHeight):
    retPath = DLPO.PATH_TEMP_MODEL_SAVE + "\\" + "ResNetModel.pth"
    graphPath = DLPO.PATH_TEMP_MODEL_RESULTS + "\\" + 'modelResult.png'

    data_transforms = getStandardModelDataTransforms(imageWidth, imageHeight)
    train_set, train_set, train_loader, valid_loader = getModelSetsAndLoaders(batchSize = batchSize, data_transforms = data_transforms)

    resnet18_model = createResnet18StandardModel() #RESNET18
    optimizer = torch.optim.Adam(resnet18_model.parameters(), lr=0.00001)
    loss = torch.nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task='multiclass', num_classes = DLPO.CLASS_QUANTITY)
    data = {"train": train_loader, "valid": valid_loader, "image_width": imageWidth, "image_height": imageHeight}
    writer = {"train": SummaryWriter(log_dir = PATH_TEMP_LOG_TRAIN),
            "valid": SummaryWriter(log_dir = PATH_TEMP_LOG_TEST)}
    
    history = train(model = resnet18_model.to("cpu"),optimizer = optimizer,
                    criterion = loss, metric = metric, data = data,
                    epochs = epochs, tb_writer = writer)
    plotModelTrainHistory(history, graphPath)
    torch.save(resnet18_model.state_dict(), retPath)
    return history, retPath, graphPath


def predictResnet18Image(modelPath, imagePath, imageWidth, imageHeight):
    predictRet = False
    model = createResnet18StandardModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    image = Image.open(imagePath)
    predictRet = makeTorchPrediction(model, image, imageWidth, imageHeight)
    return predictRet

#RESNET18 END
#########################################################


#########################################################
#VGG16
def createVGG16StandardModel():
    weights_vgg = VGG16_Weights.IMAGENET1K_V1
    #transforms_vgg = weights_vgg.transforms()
    vgg16_model = vgg16(weights=weights_vgg)
    if torch.cuda.is_available():
        vgg16_model.to("cuda")
    for param in vgg16_model.parameters():
        param.requires_grad = False
    last_layer_in_features_vgg = vgg16_model.classifier[-1].in_features
    vgg16_model.classifier[-1] = torch.nn.Linear(in_features=last_layer_in_features_vgg, out_features = DLPO.CLASS_QUANTITY)
    return vgg16_model

def trainVGG16Model(epochs, batchSize, imageWidth, imageHeight):
    retPath = DLPO.PATH_TEMP_MODEL_SAVE + "\\" + "VGG16Model.pth"
    graphPath = DLPO.PATH_TEMP_MODEL_RESULTS + "\\" + 'modelResult.png'

    data_transforms = getStandardModelDataTransforms(imageWidth, imageHeight)
    train_set, train_set, train_loader, valid_loader = getModelSetsAndLoaders(batchSize = batchSize, data_transforms = data_transforms)
    vgg16_model = createVGG16StandardModel() #VGG16

    optimizer_vgg = torch.optim.Adam(vgg16_model.parameters(), lr=0.00001)
    loss_vgg = torch.nn.CrossEntropyLoss()
    metric_vgg = torchmetrics.Accuracy(task='multiclass', num_classes = DLPO.CLASS_QUANTITY)
    data_vgg = {"train": train_loader, "valid": valid_loader, "image_width": imageWidth, "image_height": imageHeight}
    writer_vgg = {"train": SummaryWriter(log_dir = PATH_TEMP_LOG_TRAIN),
            "valid": SummaryWriter(log_dir = PATH_TEMP_LOG_TEST)}
    
    history_vgg = train(model = vgg16_model.to("cpu"),optimizer = optimizer_vgg,
                    criterion = loss_vgg, metric = metric_vgg, data = data_vgg,
                    epochs = epochs, tb_writer = writer_vgg)
    plotModelTrainHistory(history_vgg, graphPath)
    torch.save(vgg16_model.state_dict(), retPath)
    return history_vgg, retPath, graphPath

def predictVGG16Image(modelPath, imagePath, imageWidth, imageHeight):
    predictRet = False
    model = createVGG16StandardModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    image = Image.open(imagePath)
    predictRet = makeTorchPrediction(model, image, imageWidth, imageHeight)
    return predictRet
#VGG16 END
#########################################################

#########################################################
#EFFICIENTNET
def createEfficientNetStandardModel():
    weights_effnet = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    effnet_model = efficientnet_v2_l(weights=weights_effnet)
    if torch.cuda.is_available():
        effnet_model.to("cuda")
    for param in effnet_model.parameters():
        param.requires_grad = False
    last_layer_in_features_effnet = effnet_model.classifier[-1].in_features
    effnet_model.classifier[-1] = torch.nn.Linear(in_features=last_layer_in_features_effnet, out_features = DLPO.CLASS_QUANTITY)
    return effnet_model

def trainEfficientNetModel(epochs, batchSize, imageWidth, imageHeight):
    retPath = DLPO.PATH_TEMP_MODEL_SAVE + "\\" + "EfficientNetModel.pth"
    graphPath = DLPO.PATH_TEMP_MODEL_RESULTS + "\\" + 'modelResult.png'

    data_transforms = getStandardModelDataTransforms(imageWidth, imageHeight)
    train_set, train_set, train_loader, valid_loader = getModelSetsAndLoaders(batchSize = batchSize, data_transforms = data_transforms)
    effnet_model = createEfficientNetStandardModel() #VGG16

    optimizer_effnet = torch.optim.Adam(effnet_model.parameters(), lr=0.00001)
    loss_effnet = torch.nn.CrossEntropyLoss()
    metric_effnet = torchmetrics.Accuracy(task='multiclass', num_classes = DLPO.CLASS_QUANTITY)
    data_effnet = {"train": train_loader, "valid": valid_loader, "image_width": imageWidth, "image_height": imageHeight}
    writer_effnet = {"train": SummaryWriter(log_dir = PATH_TEMP_LOG_TRAIN),
            "valid": SummaryWriter(log_dir = PATH_TEMP_LOG_TEST)}

    history_effnet = train(model = effnet_model.to("cpu"), optimizer = optimizer_effnet,
                    criterion = loss_effnet, metric = metric_effnet, data = data_effnet,
                    epochs = epochs, tb_writer = writer_effnet)
    
    plotModelTrainHistory(history_effnet, graphPath)
    torch.save(effnet_model.state_dict(), retPath)
    return history_effnet, retPath, graphPath

def predictEfficientNetImage(modelPath, imagePath, imageWidth, imageHeight):
    predictRet = False
    model = createEfficientNetStandardModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    image = Image.open(imagePath)
    predictRet = makeTorchPrediction(model, image, imageWidth, imageHeight)
    return predictRet

#EFFICIENTNET END
#########################################################


#########################################################
#REGNET
def createRegNetStandardModel():
    weights_regnet = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
    transforms_regnet = weights_regnet.transforms()
    regnet_model = regnet_y_128gf(weights=weights_regnet)
    if torch.cuda.is_available():
        regnet_model.to("cuda")
    for param in regnet_model.parameters():
        param.requires_grad = False
    last_layer_in_features_regnet = regnet_model.fc.in_features
    regnet_model.fc = torch.nn.Linear(in_features=last_layer_in_features_regnet, out_features = DLPO.CLASS_QUANTITY)
    return regnet_model

def trainRegNetModel(epochs, batchSize, imageWidth, imageHeight):
    retPath = DLPO.PATH_TEMP_MODEL_SAVE + "\\" + "RegNetModel.pth"
    graphPath = DLPO.PATH_TEMP_MODEL_RESULTS + "\\" + 'modelResult.png'
    data_transforms = getStandardModelDataTransforms(imageWidth, imageHeight)
    train_set, train_set, train_loader, valid_loader = getModelSetsAndLoaders(batchSize = batchSize, data_transforms = data_transforms)
    regnet_model = createRegNetStandardModel() #REGNET
    optimizer_regnet = torch.optim.Adam(regnet_model.parameters(), lr=0.00001)
    loss_regnet = torch.nn.CrossEntropyLoss()
    metric_regnet = torchmetrics.Accuracy(task='multiclass', num_classes = DLPO.CLASS_QUANTITY)
    data_regnet = {"train": train_loader, "valid": valid_loader, "image_width": imageWidth, "image_height": imageHeight}
    writer_regnet = {"train": SummaryWriter(log_dir = PATH_TEMP_LOG_TRAIN), "valid": SummaryWriter(log_dir = PATH_TEMP_LOG_TEST)}
    history_regnet = train(model = regnet_model.to("cpu"),optimizer = optimizer_regnet,
                    criterion = loss_regnet, metric = metric_regnet, data = data_regnet,
                    epochs = epochs, tb_writer = writer_regnet)
    plotModelTrainHistory(history_regnet, graphPath)
    torch.save(regnet_model.state_dict(), retPath)
    return history_regnet, retPath, graphPath

def predictRegNetImage(modelPath, imagePath, imageWidth, imageHeight):
    predictRet = False
    model = createRegNetStandardModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    image = Image.open(imagePath)
    predictRet = makeTorchPrediction(model, image, imageWidth, imageHeight)
    return predictRet
#REGNET END
#########################################################

#########################################################
#SWIN
def createSwinStandardModel():
    weights_swin = Swin_V2_B_Weights.IMAGENET1K_V1
    transforms_swin = weights_swin.transforms()
    swin_model = swin_v2_b(weights=weights_swin)
    if torch.cuda.is_available():
        swin_model.to("cuda")
    for param in swin_model.parameters():
        param.requires_grad = False
    last_layer_in_features_swin = swin_model.head.in_features
    swin_model.head = torch.nn.Linear(in_features=last_layer_in_features_swin, out_features = DLPO.CLASS_QUANTITY)
    return swin_model

def trainSwinModel(epochs, batchSize, imageWidth, imageHeight):
    retPath = DLPO.PATH_TEMP_MODEL_SAVE + "\\" + "SwinModel.pth"
    graphPath = DLPO.PATH_TEMP_MODEL_RESULTS + "\\" + 'modelResult.png'
    data_transforms = getStandardModelDataTransforms(imageWidth, imageHeight)
    train_set, train_set, train_loader, valid_loader = getModelSetsAndLoaders(batchSize = batchSize, data_transforms = data_transforms)
    swin_model = createSwinStandardModel() #SWIN
    optimizer_swin = torch.optim.Adam(swin_model.parameters(), lr=0.00001)
    loss_swin = torch.nn.CrossEntropyLoss()
    metric_swin = torchmetrics.Accuracy(task='multiclass', num_classes = DLPO.CLASS_QUANTITY)
    data_swin = {"train": train_loader, "valid": valid_loader, "image_width": imageWidth, "image_height": imageHeight}
    writer_swin = {"train": SummaryWriter(log_dir=PATH_TEMP_LOG_TRAIN), "valid": SummaryWriter(log_dir=PATH_TEMP_LOG_TEST)}
    history_swin = train(model = swin_model.to("cpu"),optimizer = optimizer_swin,
                    criterion = loss_swin, metric = metric_swin, data = data_swin,
                    epochs = epochs, tb_writer = writer_swin)
    plotModelTrainHistory(history_swin, graphPath)
    torch.save(swin_model.state_dict(), retPath)
    return history_swin, retPath, graphPath

def predictSwinImage(modelPath, imagePath, imageWidth, imageHeight):
    predictRet = False
    model = createSwinStandardModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    image = Image.open(imagePath)
    predictRet = makeTorchPrediction(model, image, imageWidth, imageHeight)
    return predictRet
#SWIN END
#########################################################

##################################################################################
#FILE MANAGEMENT
#VER ESTO DE MOVERLO A OTRO FILE
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

def checkIfFileIsDLProgram(path):
    retBool = False
    with open(path, "r") as readFile:
        jsonDict = json.load(readFile)
    if DLPO.DL_PROGRAM_DICT_MODEL in jsonDict.keys():
        retBool = True
    return retBool
#FILE MANAGEMENT END
##################################################################################

##################################################################################
#IMAGE AUGMENTATION
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
                                     translate = (0, parameters[DLPO.AUGMENT_RANDOM_AFFINE_CONFIG_TRANSLATE]),
                                     scale = (0, parameters[DLPO.AUGMENT_RANDOM_AFFINE_CONFIG_SCALE]),
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

def getAugmentVariableNames(augment):
    variableNames = []
    if augment == DLPO.AUGMENT_OPTIONS_RESIZE:
        variableNames.append(DLPO.augmentResizeNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentResizeNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_RESIZE:
        variableNames.append(DLPO.augmentRandomResizeNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentRandomResizeNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_CROP:
        variableNames.append(DLPO.augmentRandomCropNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentRandomCropNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
        variableNames.append(DLPO.augmentRandomCropNames[DLPO.AUGMENT_CONFIG_VARIABLES_3])
    elif augment == DLPO.AUGMENT_OPTIONS_CENTER_CROP:
        variableNames.append(DLPO.augmentCenterCropNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentCenterCropNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_HORIZONTAL_FLIP:
        variableNames.append(DLPO.augmentRandomHorizontalFlipNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_VERTICAL_FLIP:
        variableNames.append(DLPO.augmentRandomVerticalFlipNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
    elif augment == DLPO.AUGMENT_OPTIONS_PAD:
        variableNames.append(DLPO.augmentPadNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_ROTATION:
        variableNames.append(DLPO.augmentRandomRotationNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_AFFINE:
        variableNames.append(DLPO.augmentRandomAffineNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentRandomAffineNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
        variableNames.append(DLPO.augmentRandomAffineNames[DLPO.AUGMENT_CONFIG_VARIABLES_3])
        variableNames.append(DLPO.augmentRandomAffineNames[DLPO.AUGMENT_CONFIG_VARIABLES_4])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_PERSPECTIVE:
        variableNames.append(DLPO.augmentRandomPerspectiveNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentRandomPerspectiveNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
    elif augment == DLPO.AUGMENT_OPTIONS_COLOR_JITTER:
        variableNames.append(DLPO.augmentColorJitterNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentColorJitterNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
        variableNames.append(DLPO.augmentColorJitterNames[DLPO.AUGMENT_CONFIG_VARIABLES_3])
        variableNames.append(DLPO.augmentColorJitterNames[DLPO.AUGMENT_CONFIG_VARIABLES_4])
    elif augment == DLPO.AUGMENT_OPTIONS_GAUSSIAN_BLUR:
        variableNames.append(DLPO.augmentGaussianBlurNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
        variableNames.append(DLPO.augmentGaussianBlurNames[DLPO.AUGMENT_CONFIG_VARIABLES_2])
    elif augment == DLPO.AUGMENT_OPTIONS_RANDOM_INVERT:
        variableNames.append(DLPO.augmentRandomInvertNames[DLPO.AUGMENT_CONFIG_VARIABLES_1])
    return variableNames

#IMAGE AUGMENTATION END
##################################################################################

##################################################################################
#MULTI-MODEL FUNCTIONS
def createImageSplitFolders():
    path = PATH_TEMP_SPLIT_IMAGES
    try:
        os.mkdir(path + "\\train")
    except:
        pass
    try:
        os.mkdir(path + "\\test")
    except:
        pass
    try:
        os.mkdir(path + "\\train\\" + DLPO.DL_CLASS_OK)
    except:
        pass
    try:
        os.mkdir(path + "\\train\\" + DLPO.DL_CLASS_NOK)
    except:
        pass
    try:
        os.mkdir(path + "\\test\\" + DLPO.DL_CLASS_OK)
    except:
        pass
    try:
        os.mkdir(path + "\\test\\" + DLPO.DL_CLASS_NOK)
    except:
        pass

def prepareTorchModelFiles(okPath, nokPath, testTrainSplit): #ESTA FUNCION QUEDO PRACTICAMENTE IGUAL A LA DE YOLOV8 - VER DE UNIFICAR
    if okPath == None or nokPath == None:
        print("Error: Image paths not found")
        return #return if paths have not been set
    createImageSplitFolders()
    #Delete existing images in train folders
    deletePaths = [PATH_TEMP_SPLIT_IMAGES_TRAIN + "\\OK", PATH_TEMP_SPLIT_IMAGES_TRAIN + "\\NOK", PATH_TEMP_SPLIT_IMAGES_TEST + "\\OK", PATH_TEMP_SPLIT_IMAGES_TEST + "\\NOK"]
    for folderPath in deletePaths:
        deleteFolderFiles(folderPath)
    #Split and copy the images
    threshold = testTrainSplit / 100 #Write test Train Split as a float
    for img in os.listdir(okPath): #Split OK Images
        if os.path.isfile(okPath + "\\" + img):
            if random.uniform(0,1) > threshold:
                shutil.copy(okPath + "\\" + img, PATH_TEMP_SPLIT_IMAGES_TRAIN + "\\OK")
            else:
                shutil.copy(okPath + "\\" + img, PATH_TEMP_SPLIT_IMAGES_TEST + "\\OK")
    for img in os.listdir(nokPath): #Split NOK Images
        if os.path.isfile(nokPath + "\\" + img):
            if random.uniform(0,1) > threshold:
                shutil.copy(nokPath + "\\" + img, PATH_TEMP_SPLIT_IMAGES_TRAIN + "\\NOK")
            else:
                shutil.copy(nokPath + "\\" + img, PATH_TEMP_SPLIT_IMAGES_TEST + "\\NOK")

def train(model, optimizer, criterion, metric, data, epochs, tb_writer=None):
    train_loader = data["train"]
    valid_loader = data["valid"]
    train_writer = tb_writer["train"]
    valid_writer = tb_writer["valid"]
    if tb_writer:
        train_writer.add_graph(model, torch.zeros((1, 3, data["image_width"], data["image_height"])))
        valid_writer.add_graph(model, torch.zeros((1, 3, data["image_width"], data["image_height"])))
    if torch.cuda.is_available():
        model.to("cuda")
        metric.to("cuda")
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        for train_data, train_target in train_loader:
            if torch.cuda.is_available():
                train_data = train_data.to("cuda")
                train_target = train_target.to("cuda")
            optimizer.zero_grad()
            output = model(train_data.float())
            loss = criterion(output, train_target)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            accuracy = metric(output, train_target)
            epoch_train_accuracy += accuracy.item()
        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_accuracy = epoch_train_accuracy / len(train_loader)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_accuracy)
        
        model.eval()
        epoch_valid_loss = 0.0
        epoch_valid_accuracy = 0.0
        for valid_data, valid_target in valid_loader:
            if torch.cuda.is_available():
                valid_data = valid_data.to("cuda")
                valid_target = valid_target.to("cuda")
            output = model(valid_data.float())
            epoch_valid_loss += criterion(output, valid_target).item()
            epoch_valid_accuracy += metric(output, valid_target).item()  
        epoch_valid_loss = epoch_valid_loss / len(valid_loader)
        epoch_valid_accuracy = epoch_valid_accuracy / len(valid_loader)
        valid_loss.append(epoch_valid_loss)
        valid_acc.append(epoch_valid_accuracy)
        print("Epoch: {}/{} - Train loss {:.6f} - Train Accuracy {:.6f} - Valid Loss {:.6f} - Valid Accuracy {:.6f}".format(
        epoch+1, epochs, epoch_train_loss, epoch_train_accuracy, epoch_valid_loss, epoch_valid_accuracy))
        if tb_writer:
            train_writer.add_scalar("loss", epoch_train_loss, epoch)
            valid_writer.add_scalar("loss", epoch_valid_loss, epoch)
            train_writer.add_scalar("accuracy", epoch_train_accuracy, epoch)
            valid_writer.add_scalar("accuracy", epoch_valid_accuracy, epoch)
            train_writer.flush()
            valid_writer.flush()
    history = {}
    history["train_loss"] = train_loss
    history["train_acc"] = train_acc
    history["valid_loss"] = valid_loss
    history["valid_acc"] = valid_acc
    return history

def getStandardModelDataTransforms(imageWidth, imageHeight):
    data_transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(size=(imageWidth, imageHeight)),
                        torchvision.transforms.ToTensor()
                        ])
    return data_transforms

def getModelSetsAndLoaders(batchSize, data_transforms):
    train_set = torchvision.datasets.ImageFolder(root = PATH_TEMP_SPLIT_IMAGES_TRAIN, transform=data_transforms)
    valid_set = torchvision.datasets.ImageFolder(root = PATH_TEMP_SPLIT_IMAGES_TEST, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batchSize, shuffle=True)
    return train_set, valid_set, train_loader, valid_loader

def plotModelTrainHistory(history, path):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(history["train_loss"]) 
    axs[0].plot(history["valid_loss"]) 
    axs[0].title.set_text('Train vs Valid Error') 
    axs[0].legend(['Train', 'Valid'])  
    axs[1].plot(history["train_acc"]) 
    axs[1].plot(history["valid_acc"]) 
    axs[1].title.set_text('Train vs Valid Accuracy') 
    axs[1].legend(['Train', 'Valid'])
    plt.savefig(path)

def makeTorchPrediction(model, image, imageWidth, imageHeight):
    predictRet = False
    data_transforms = getStandardModelDataTransforms(imageWidth, imageHeight)
    imageTransformed = data_transforms(image)
    prediction = model(imageTransformed.unsqueeze(0)).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    if class_id == 1: #Class is identified as OK
        predictRet = True
    return predictRet
#MULTI-MODEL FUNCTIONS END
##################################################################################
