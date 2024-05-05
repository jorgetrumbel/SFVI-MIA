import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem
from PyQt5.QtCore import QModelIndex, Qt, QRunnable, QThreadPool, pyqtSignal
from PyQt5.uic import loadUi
from PIL import Image as im

from DialogCommandSelection import DialogCommandSelection
from DialogProgramSelection import DialogProgramSelection
from DialogTrainOutput import DialogTrainOutput
from DialogAugmentSelection import DialogAugmentSelection
from DialogGeneralConfig import DialogGeneralConfig
from DialogCameraConfig import DialogCameraConfig
from DialogCaptureView import DialogCaptureView

import VisionProgramOptions as VPO
import DeepLearningProgramOptions as DLPO
from VisionProgram import ProgramStructure
import VisionProgram as VP
import DeepLearningModule as DLM
import DialogTrainOutput as DTO
import UtilitiesModule as UM
import ProgramCommonPaths as PCP
import ProgramConfigOptions as PCO


if UM.isRPi():
    #Working on RPi
    import GPIOModule as IO
    import CameraModule as CM
else:
    import GPIOModuleDummy as IO
    import CameraModuleDummy as CM
    import CameraModulePC as CMPC

import json #FOR DEBUGGING
import cv2 as cv #FOR DEBUGGING


stackOptionsNames = ("stackCaptureOptions", "StackFilterOptions", "stackFeatureDetectionOptions", "stackDrawOptions", "stackMeasurementOptions")
STACK_OPTIONS_CAPTURE_WIDGET_NAME = stackOptionsNames[0]
STACK_OPTIONS_FILTER_WIDGET_NAME = stackOptionsNames[1]
STACK_OPTIONS_FEATURE_DETECTION_WIDGET_NAME = stackOptionsNames[2]
STACK_OPTIONS_DRAW_OPTIONS_WIDGET_NAME = stackOptionsNames[3]
STACK_OPTIONS_MEASUREMENT_OPTIONS_WIDGET_NAME = stackOptionsNames[4]

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(300, 100, 800, 600)
        self.setWindowTitle('SFVI - Sistema Flexible de Visión Industrial')
        self.setWindowIcon(QIcon('images/apple.PNG'))
        self.setupScreens()
        self.show()

    def setupScreens(self):
        loadUi("ui/SFVI.ui", self)
        self.setupScreenLogic()
        self.stackWidget.setCurrentWidget(self.ScreenMonitorMain) #Set initial screen
    
    def setupScreenLogic(self):
        self.ScreenMonitorMainLogic()
        self.ScreenProgrammingMainLogic()
        self.ScreenProgramEditorLogic()
        self.ScreenDLProgramEditorLogic()


    def resetProgramCounters(self):
        self.programPicturesTaken = 0
        self.programNOKPictures = 0

    programOnlineStatus = PCO.PROGRAM_ONLINE_STATUS_OFFLINE
    selectedProgram = None
    selectedProgramName = None
    selectedProgramType = None
    programPicturesTaken = 0
    programNOKPictures = 0
    camera = CM.Camera()
    boardIO = IO.IO()
    programTriggerSignal = pyqtSignal()

    #########################################################
    #ScreenMonitorMain
    def ScreenMonitorMainLogic(self):
        self.updateProgramStatusForm()
        self.programTriggerSignal.connect(self.triggerProgramRun)
        self.boardIO.setTriggerPinFunc(self.programTrigger) #MOVER DE ACA?? REVISAR
        self.buttonChangeToProgrammingMain.clicked.connect(self.goToScreenProgrammingMain)
        self.buttonSelectProgramScreenMonitorMain.clicked.connect(self.getProgramFileName)
        self.buttonCounterScreenMonitorMain.clicked.connect(self.triggerProgramRun) #CORREGIR - NO VA EN ESTE BOTON
        self.buttonConfigScreenMonitorMain.clicked.connect(self.configButtonAction)
        self.buttonCameraConfigurationScreenMonitorMain.clicked.connect(self.cameraConfigButtonAction)

    def goToScreenProgrammingMain(self):
        self.stackWidget.setCurrentWidget(self.ScreenProgrammingMain)
        self.loadListView()
        
    def getProgramFileName(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self,"Select File", "","All Files (*);;Python Files (*.py)", options=options)
        if file:
            self.selectedProgram = file
            self.selectedProgramName = UM.getFileNameFromPath(file)
            if VP.checkIfFileIsVisionProgram(file): #Classic vision program selected
                self.visionProgramStructure.loadProgram(file)
                self.selectedProgramType = VPO.VISION_PROGRAM_TYPES_CLASSIC
            elif DLM.checkIfFileIsDLProgram(file):
                self.selectedProgramType = VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING
                self.DLmodel.loadProgram(file)
        self.updateProgramStatusForm()
    
    def configButtonAction(self):
        generalConfigDialog = DialogGeneralConfig(self)
        generalConfigDialog.exec()
        self.programOnlineStatus = generalConfigDialog.getProgramStatus()
        if generalConfigDialog.checkIfResetWasPressed():
            self.resetProgramCounters()
        self.updateProgramStatusForm()

    def captureAction(self):
        image = None
        if UM.isRPi():
            #RPI
            image = self.camera.takeArray()
        else:
            #Windows
            image = CMPC.takePicturePC()
        captureViewDialog = DialogCaptureView(self)
        captureViewDialog.setImage(image)
        captureViewDialog.exec()
        return image

    def cameraConfigButtonAction(self):
        cameraConfigDialog = DialogCameraConfig(self)
        cameraConfigDialog.setCaptureFunc(self.captureAction)
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

    def updateProgramStatusForm(self):
        self.lineEditProgramStatusScreenMonitorMain.setText(self.programOnlineStatus)
        self.lineEditProgramNameScreenMonitorMain.setText(self.selectedProgramName)
        self.lineEditPicturesTakenScreenMonitorMain.setText(str(self.programPicturesTaken))
        self.lineEditNOKPicturesScreenMonitorMain.setText(str(self.programNOKPictures))

    def programTrigger(self):
        self.programTriggerSignal.emit()

    def triggerProgramRun(self):
        self.boardIO.setBusyPinFunc(True)
        if self.selectedProgramType == VPO.VISION_PROGRAM_TYPES_CLASSIC:
            image, data, dataType, programIndividualResults, programResult = self.visionProgramStructure.runProgram(True)
        elif self.selectedProgramType == VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file, _ = QFileDialog.getOpenFileName(self,"Select File", "","All Files (*);;Python Files (*.py)", options=options)
            #CORREGIR - LA IMAGEN TENDRIA QUE SACARSE, NO ELEGIR DE LOS ARCHIVOS
            programResult, image = self.DLmodel.loadedModelPredict(imagePath = file)
        self.setImageScreenMonitorMain(image)
        self.programPicturesTaken = self.programPicturesTaken + 1
        if programResult == False:
            self.programNOKPictures = self.programNOKPictures + 1
            self.boardIO.NOKPinFunc()
        else:
            self.boardIO.OKPinFunc()
        self.updateProgramStatusForm()
        self.boardIO.setBusyPinFunc(False)
        
    def setImageScreenMonitorMain(self, image):
        try:
            tempImagePath = PCP.PATH_TEMP_VISION_PROGRAM_IMAGE
            data = im.fromarray(image)
            data.save(tempImagePath)
            pixmap = QPixmap(tempImagePath)
            self.labelImageScreenMonitorMain.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")

    #End ScreenMonitorMain
    #########################################################

    #########################################################
    #ScreenProgrammingMain
    def ScreenProgrammingMainLogic(self):
        self.loadListView()
        self.listViewScreenProgrammingMain.clicked.connect(self.listViewClicked)
        self.buttonChangeToMonitorMain.clicked.connect(self.goToScreenMonitorMain)
        self.buttonNewProgramScreenProgrammingMain.clicked.connect(self.launchNewProgramDialog)
        self.buttonEditProgramScreenProgrammingMain.clicked.connect(self.editProgramButtonAction)
        self.buttonCopyProgramScreenProgrammingMain.clicked.connect(self.copyProgramButtonAction)
        self.buttonEraseProgramScreenProgrammingMain.clicked.connect(self.deleteProgramButtonAction)

    def goToScreenProgramEditor(self):
        self.stackWidget.setCurrentWidget(self.ScreenProgramEditor)

    def goToScreenMonitorMain(self):
        self.stackWidget.setCurrentWidget(self.ScreenMonitorMain)

    def goToScreenDLProgramEditor(self):
        self.stackWidget.setCurrentWidget(self.ScreenDLProgramEditor)

    def loadListView(self):
        #Setup listView itemModel
        self.programListItemModel = QStandardItemModel()
        self.listViewScreenProgrammingMain.setModel(self.programListItemModel)
        #Get paths for items
        paths = UM.searchSubfoldersForFilesEndingIn(endString = ".json", path = PCP.PATH_SAVED_PROGRAMS)
        for path in paths:
            item = QStandardItem(str(path))
            self.programListItemModel.appendRow(item)

    def listViewClicked(self, index):
        #Get configuration from currently selected index and show it in the label
        selectedIndex = self.listViewScreenProgrammingMain.selectedIndexes()[0]
        selectedItem = self.programListItemModel.itemFromIndex(selectedIndex)
        programString = UM.getJsonStringFromFile(selectedItem.text())
        self.labelScreenProgrammingMain.setText(programString)
        
    def editProgramButtonAction(self):
        #Get selected index from list
        selectedIndex = self.listViewScreenProgrammingMain.selectedIndexes()[0]
        selectedItem = self.programListItemModel.itemFromIndex(selectedIndex)
        programPath = selectedItem.text()
        if DLM.checkIfFileIsDLProgram(programPath): #DL Program selected
            self.DLmodel.loadProgram(programPath)
            self.loadProgramToDLTreeViewScreenDLProgramEditor()
            self.goToScreenDLProgramEditor()
        elif VP.checkIfFileIsVisionProgram(programPath): #Classic vision program selected
            self.visionProgramStructure.loadProgram(programPath)
            self.loadProgramToTreeViewScreenProgramEditor()
            self.goToScreenProgramEditor()

    def copyProgramButtonAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, filter = QFileDialog.getSaveFileName(self, caption = "Save File", directory = PCP.PATH_SAVED_PROGRAMS, filter = "Program Files (*.json);;All Files (*)", options = options)
        #Get selected index from list
        selectedIndex = self.listViewScreenProgrammingMain.selectedIndexes()[0]
        selectedItem = self.programListItemModel.itemFromIndex(selectedIndex)
        pathToCopyFile = selectedItem.text()
        UM.copyProgram(file, pathToCopyFile)
        self.loadListView()

    def deleteProgramButtonAction(self):
        #Show confirmation Dialog
        buttonReply = QMessageBox.question(self, 'Delete Program', "Are you sure you want to delete the selected program?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            #Yes was clicked, delete program
            #Get selected index from list
            selectedIndex = self.listViewScreenProgrammingMain.selectedIndexes()[0]
            selectedItem = self.programListItemModel.itemFromIndex(selectedIndex)
            pathToDeleteFile = selectedItem.text()
            UM.deleteProgram(pathToDeleteFile)
            self.loadListView()
        else:
            #No was clicked, do nothing
            pass 


    def launchNewProgramDialog(self):
        #Launch dialog
        programSelectDialog = DialogProgramSelection(self)
        programSelectDialog.exec()
        programReturnString = programSelectDialog.getReturnString()
        programTypeString = programSelectDialog.getProgramType()
        if programTypeString == VPO.VISION_PROGRAM_TYPES_CLASSIC:
            self.launchNewVisionProgram()
            self.goToScreenProgramEditor()
        elif programTypeString == VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING:
            #AGREGAR QUE SE CREE UN NUEVO PROGRAMA - SE VACIE LA INFO EN PROGRAM EDITOR
            self.launchNewDLProgram()
            self.DLmodel.setSelectedModel(programReturnString)
            self.goToScreenDLProgramEditor()

    programListItemModel = None

    #End ScreenProgrammingMain
    #########################################################

    #########################################################
    #ScreenProgramEditor
    def ScreenProgramEditorLogic(self):
        #self.setImageScreenProgramEditor()
        self.loadTreeView()
        self.visionProgramStructure.getCamera(self.camera)
        self.treeViewScreenProgramEditor.clicked.connect(self.treeViewClicked)
        self.buttonExitScreenProgramEditor.clicked.connect(self.exitVisionProgramEditorButtonAction)
        self.buttonAddCommandScreenProgramEditor.clicked.connect(self.addCommandToTree)
        self.buttonRunScreenProgramEditor.clicked.connect(self.runVisionProgram)
        self.buttonDeleteCommandScreenProgramEditor.clicked.connect(self.deleteCommandFromTree)
        self.buttonFeatureDetectionTemplate.clicked.connect(lambda: self.visionProgramStructure.selectTemplate(self.getSelectedInstructionName(), self.getSelectedInstructionParentName()))
        self.buttonFilterCropArea.clicked.connect(lambda: self.visionProgramStructure.selectCropArea(self.getSelectedInstructionName(), self.getSelectedInstructionParentName()))
        self.buttonCaptureSelectFile.clicked.connect(self.getCaptureFileName)
        self.buttonSaveProgramScreenProgramEditor.clicked.connect(self.saveVisionProgramButtonAction)
        self.buttonCaptureCameraConfig.clicked.connect(self.cameraConfigProgramEditorButtonAction)

    def setImageScreenProgramEditor(self, image):
        try:
            tempImagePath = PCP.PATH_TEMP_VISION_PROGRAM_IMAGE
            data = im.fromarray(image)
            data.save(tempImagePath)
            pixmap = QPixmap(tempImagePath)
            self.labelImageScreenProgramEditor.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")

    def loadTreeView(self):
        #Configure QTreeView
        self.itemModel = QStandardItemModel()
        self.treeViewScreenProgramEditor.setModel(self.itemModel)
        self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackCaptureOptions)
    
    def launchNewVisionProgram(self):
        self.itemModel.clear() #clear treeView itemModel
        parentItem = self.itemModel.invisibleRootItem()
        self.visionProgramStructure.clearProgram()
        self.labelImageScreenProgramEditor.clear()
        self.visionProgramStructure.addInstruction("File Select1", parentItem.text(), VPO.CAPTURE_OPTIONS_FILE_SELECT)
        self.addInstructionToTreeView("File Select1", parentItem.text()) #CORREGIR

    def loadProgramToTreeViewScreenProgramEditor(self):
        self.itemModel.clear() #clear treeView itemModel
        names, parents, config = self.visionProgramStructure.getProgramAttributes() #Get program
        for index, name in enumerate(names):
            self.addInstructionToTreeView(name, parents[index])
            instructionType = self.visionProgramStructure.getInstructionType(name)
            self.updateStackedWidgetScreenProgramEditor(instructionType, config[index])
            self.treeViewClicked(self.treeIndex)

    def addInstructionToTreeView(self, instructionName, parentName):
        if parentName == "":
            parentItem = self.itemModel.invisibleRootItem()
        else:
            parentItem = self.itemModel.findItems(parentName, flags = Qt.MatchRecursive)
            if len(parentItem) > 0:
                parentItem = parentItem[0]
        if parentItem:
            item = QStandardItem(instructionName)
            parentItem.appendRow(item)
            self.treeIndex = item.index()
            self.treeViewScreenProgramEditor.setCurrentIndex(self.treeIndex)
            self.treeViewClicked(item.index())

    def treeViewClicked(self, index):
        #Get configuration from currently selected tree command and pass it to vision program
        previousItem = self.itemModel.itemFromIndex(self.treeIndex)
        configuration = self.getInstructionConfigurationFromTree()
        if previousItem != None:
            self.visionProgramStructure.changeInstructionConfiguration(previousItem.text(), configuration)
        self.treeIndex = index #Update treeIndex
        #Update currently displayed widget according to type of instruction selected
        item = self.itemModel.itemFromIndex(index)
        instructionType = self.visionProgramStructure.getInstructionType(item.text())
        instructionConfiguration = self.visionProgramStructure.getInstructionConfiguration(item.text())
        self.updateInstructionVariableNames(instructionType = instructionType)
        self.updateStackedWidgetScreenProgramEditor(instructionType, instructionConfiguration)

    def updateStackedWidgetScreenProgramEditor(self, instructionType, instructionConfiguration):
        if instructionType in VPO.captureOptions:
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackCaptureOptions)
            self.loadStackCaptureOptions(instructionConfiguration)
        elif instructionType in VPO.filterOptions:
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.StackFilterOptions)
            self.loadStackFilterOptions(instructionConfiguration)
        elif instructionType in VPO.featureDetectionOptions:
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackFeatureDetectionOptions)
            self.loadStackFeatureDetectionOptions(instructionConfiguration)
        elif instructionType in VPO.drawOptions:
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackDrawOptions)
            self.loadStackDrawOptions(instructionConfiguration)
        elif instructionType in VPO.measurementOptions:
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackMeasurementOptions)
            self.loadStackMeasurementOptions(instructionConfiguration)

    def loadStackFilterOptions(self, configuration):
        self.spinBoxKernelRows.setValue(configuration[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS])
        self.spinBoxKernelColumns.setValue(configuration[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS])
        self.spinBoxIterations.setValue(configuration[VPO.FILTER_CONFIGURATIONS_ITERATIONS])
        self.spinBoxThreshold.setValue(configuration[VPO.FILTER_CONFIGURATIONS_THRESHOLD])
        self.spinBoxThreshold2.setValue(configuration[VPO.FILTER_CONFIGURATIONS_THRESHOLD2])

    def loadStackCaptureOptions(self, configuration):
        pass
    
    def loadStackFeatureDetectionOptions(self, configuration):
        self.spinBoxFeatureDetectionVariable1.setValue(configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1])
        self.spinBoxFeatureDetectionVariable2.setValue(configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2])
        self.spinBoxFeatureDetectionVariable3.setValue(configuration[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3])

    def loadStackDrawOptions(self, configuration):
        self.spinBoxDrawOptionsVariable1.setValue(configuration[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_1])
        self.spinBoxDrawOptionsVariable2.setValue(configuration[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_2])
        self.spinBoxDrawOptionsVariable3.setValue(configuration[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_3])

    def loadStackMeasurementOptions(self, configuration):
        self.spinBoxMeasurementOptionsVariable1.setValue(configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_1])
        self.spinBoxMeasurementOptionsVariable2.setValue(configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_2])
        self.spinBoxMeasurementOptionsVariable3.setValue(configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_3])
        self.spinBoxMeasurementOptionsVariable4.setValue(configuration[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_4])

    def addCommandToTree(self):
        #Launch dialog
        commandSelectDialog = DialogCommandSelection(self)
        commandSelectDialog.exec()
        dialogReturnString = commandSelectDialog.getReturnString()
        parentItem = self.itemModel.itemFromIndex(self.treeIndex)
        instructionType = dialogReturnString
        instructionCounter = 1
        dialogReturnStringTemp = dialogReturnString + str(instructionCounter)
        while(self.visionProgramStructure.checkInstrucionName(dialogReturnStringTemp)):
            instructionCounter = instructionCounter + 1
            dialogReturnStringTemp = dialogReturnString + str(instructionCounter)
        dialogReturnString = dialogReturnStringTemp
        if parentItem != None:
            parentItemText = parentItem.text()
        else:
            parentItemText = ""
            parentItem = self.itemModel.invisibleRootItem()
        self.visionProgramStructure.addInstruction(dialogReturnString, parentItemText, instructionType)
        item = QStandardItem(dialogReturnString)
        parentItem.appendRow(item)
        #Select newly created index on treeView
        self.treeViewClicked(item.index())
        self.treeViewScreenProgramEditor.setCurrentIndex(item.index())

    def deleteCommandFromTree(self):
        selectedIndex = QModelIndex(self.treeViewScreenProgramEditor.selectedIndexes()[0])
        selectedItem = self.itemModel.itemFromIndex(selectedIndex)
        self.visionProgramStructure.removeInstruction(selectedItem.text())
        self.itemModel.removeRow(selectedIndex.row(), selectedIndex.parent())
        self.treeIndex = selectedIndex.parent()
        item = self.itemModel.itemFromIndex(self.treeIndex)
        if item != None:
            instructionType = self.visionProgramStructure.getInstructionType(item.text())
            instructionConfiguration = self.visionProgramStructure.getInstructionConfiguration(item.text())
            self.updateStackedWidgetScreenProgramEditor(instructionType, instructionConfiguration)
            self.treeViewClicked(self.treeIndex)
        self.treeViewScreenProgramEditor.setCurrentIndex(self.treeIndex)
         
    def getInstructionConfigurationFromTree(self):
        stackCurrentWidgetName = self.stackedWidgetScreenProgramEditor.currentWidget().objectName()
        if stackCurrentWidgetName == STACK_OPTIONS_CAPTURE_WIDGET_NAME:
            instructionData = self.getStackCaptureConfiguration()
        elif stackCurrentWidgetName == STACK_OPTIONS_FILTER_WIDGET_NAME:
            instructionData = self.getStackFilterConfiguration()
        elif stackCurrentWidgetName == STACK_OPTIONS_FEATURE_DETECTION_WIDGET_NAME:
            instructionData = self.getStackFeatureDetectionConfiguration()
        elif stackCurrentWidgetName == STACK_OPTIONS_DRAW_OPTIONS_WIDGET_NAME:
            instructionData = self.getStackDrawOptionsConfiguration()
        elif stackCurrentWidgetName == STACK_OPTIONS_MEASUREMENT_OPTIONS_WIDGET_NAME:
            instructionData = self.getStackMeasurementOptionsConfiguration()
        return instructionData

    def updateInstructionVariableNames(self, instructionType):
        variableNames = []
        if instructionType in VPO.featureDetectionOptions:
            labels = [self.labelFeatureDetectionVariable1, self.labelFeatureDetectionVariable2, self.labelFeatureDetectionVariable3]
            spinBoxes = [self.spinBoxFeatureDetectionVariable1, self.spinBoxFeatureDetectionVariable2, self.spinBoxFeatureDetectionVariable3]
        elif instructionType in VPO.drawOptions:
            labels = [self.labelDrawOptionsVariable1, self.labelDrawOptionsVariable2, self.labelDrawOptionsVariable3]
            spinBoxes = [self.spinBoxDrawOptionsVariable1, self.spinBoxDrawOptionsVariable2, self.spinBoxDrawOptionsVariable3]
        elif instructionType in VPO.measurementOptions:
            labels = [self.labelMeasurementOptionsVariable1, self.labelMeasurementOptionsVariable2, self.labelMeasurementOptionsVariable3, self.labelMeasurementOptionsVariable4]
            spinBoxes = [self.spinBoxMeasurementOptionsVariable1, self.spinBoxMeasurementOptionsVariable2, self.spinBoxMeasurementOptionsVariable3, self.spinBoxMeasurementOptionsVariable4]
        else:
            return #Exit function if instruction type is not correct
        variableNames = VP.getInstructionVariableNames(instructionType)
        for index in range(len(labels)):
            labels[index].setText("N/A")
            if index < len(variableNames):
                labels[index].setText(variableNames[index])
                spinBoxes[index].setEnabled(True)
            else:
                spinBoxes[index].setEnabled(False)

    def getStackCaptureConfiguration(self):
        instructionData = {}
        instructionData[VPO.CAPTURE_CONFIGURATIONS_NAME] = self.lineEditCaptureName.text()
        return instructionData
    
    def getStackFilterConfiguration(self):
        instructionData = {}
        instructionData[VPO.FILTER_CONFIGURATIONS_NAME] = self.lineEditFilterName.text()
        instructionData[VPO.FILTER_CONFIGURATIONS_KERNEL_ROWS] = self.spinBoxKernelRows.value()
        instructionData[VPO.FILTER_CONFIGURATIONS_KERNEL_COLUMNS] = self.spinBoxKernelColumns.value()
        instructionData[VPO.FILTER_CONFIGURATIONS_ITERATIONS] = self.spinBoxIterations.value()
        instructionData[VPO.FILTER_CONFIGURATIONS_THRESHOLD] = self.spinBoxThreshold.value()
        instructionData[VPO.FILTER_CONFIGURATIONS_THRESHOLD2] = self.spinBoxThreshold2.value()
        return instructionData
    
    def getStackFeatureDetectionConfiguration(self):
        instructionData = {}
        instructionData[VPO.FEATURE_DETECTION_CONFIGURATIONS_NAME] = self.lineEditFeatureDetectionName.text()
        instructionData[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1] = self.spinBoxFeatureDetectionVariable1.value()
        instructionData[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2] = self.spinBoxFeatureDetectionVariable2.value()
        instructionData[VPO.FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3] = self.spinBoxFeatureDetectionVariable3.value()
        return instructionData

    def getStackDrawOptionsConfiguration(self):
        instructionData = {}
        instructionData[VPO.DRAW_OPTIONS_CONFIGURATIONS_NAME] = self.lineEditDrawOptionsName.text()
        instructionData[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_1] = self.spinBoxDrawOptionsVariable1.value()
        instructionData[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_2] = self.spinBoxDrawOptionsVariable2.value()
        instructionData[VPO.DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_3] = self.spinBoxDrawOptionsVariable3.value()
        return instructionData
    
    def getStackMeasurementOptionsConfiguration(self):
        instructionData = {}
        instructionData[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_NAME] = self.lineEditMeasurementOptionsName.text()
        instructionData[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_1] = self.spinBoxMeasurementOptionsVariable1.value()
        instructionData[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_2] = self.spinBoxMeasurementOptionsVariable2.value()
        instructionData[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_3] = self.spinBoxMeasurementOptionsVariable3.value()
        instructionData[VPO.MEASUREMENT_OPTIONS_CONFIGURATIONS_VARIABLE_4] = self.spinBoxMeasurementOptionsVariable4.value()
        return instructionData

    def getSelectedInstructionName(self):
        currentItem = self.itemModel.itemFromIndex(self.treeIndex)
        name = currentItem.text()
        return name

    def getSelectedInstructionParentName(self):
        currentItem = self.itemModel.itemFromIndex(self.treeIndex)
        name = currentItem.parent().text()
        return name

    def updateTableView(self, data, dataType):
        headers = [0]
        tableModel = MyTableModel(data = data)
        if dataType == VPO.FEATURE_DETECTION_OPTIONS_CONTOURS:
            headers = VPO.FEATURE_MEASUREMENT_CONTOURS_NAMES
        elif dataType == VPO.FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH:
            headers = VPO.FEATURE_MEASUREMENT_TEMPLATE_MATCHING_NAMES
        elif dataType == VPO.FEATURE_DETECTION_OPTIONS_HOUGH:
            headers = VPO.FEATURE_MEASUREMENT_HOUGH_NAMES
        elif dataType == VPO.FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC or dataType == VPO.FEATURE_DETECTION_OPTIONS_LINE_DETECTOR:
            headers = VPO.FEATURE_MEASUREMENT_HOUGH_PROBABILISTIC_NAMES
        tableModel.getHeaders(headers, None)
        self.tableViewScreenProgramEditor.setModel(tableModel)

    def runVisionProgram(self):
        currentItem = self.itemModel.itemFromIndex(self.treeIndex)
        configuration = self.getInstructionConfigurationFromTree()
        self.visionProgramStructure.changeInstructionConfiguration(currentItem.text(), configuration)
        image, data, dataType, programIndividualResults, programResult = self.visionProgramStructure.runProgram(False, currentItem.text())
        self.updateTableView(data, dataType)
        self.setImageScreenProgramEditor(image)

    def getCaptureFileName(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self,"Select File", "","All Files (*);;Python Files (*.py)", options=options)
        if file:
            currentItem = self.itemModel.itemFromIndex(self.treeIndex)
            self.visionProgramStructure.getCaptureFileName(currentItem.text(), file)

    def cameraConfigProgramEditorButtonAction(self):
        cameraConfigDialog = DialogCameraConfig(self)
        currentItem = self.itemModel.itemFromIndex(self.treeIndex)
        previousCameraConfig = self.visionProgramStructure.getCameraConfig(currentItem.text())
        cameraConfigDialog.loadCameraConfigTransformed(previousCameraConfig)        
        cameraConfigDialog.exec()
        cameraOptions = cameraConfigDialog.getFormsValuesTransformed()
        if cameraConfigDialog.getDialogResult():
            #Ok Clicked - Change camera config
            self.visionProgramStructure.changeCameraConfigurations(currentItem.text(), cameraOptions)
            if UM.isRPi():
                self.camera.loadControlConfig(cameraOptions)

    def saveVisionProgramButtonAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, filter = QFileDialog.getSaveFileName(self, caption = "Save File", directory = PCP.PATH_SAVED_PROGRAMS, filter = "Program Files (*.json);;All Files (*)", options = options)
        self.visionProgramStructure.saveProgram(file)
        
    def exitVisionProgramEditorButtonAction(self):
        self.visionProgramStructure.clearProgram()
        self.labelImageScreenProgramEditor.clear()
        self.updateTableView([[0]], None)
        self.goToScreenProgrammingMain()

    #End ScreenProgramEditor
    #########################################################

    #########################################################
    #ScreenDLProgramEditor
    def ScreenDLProgramEditorLogic(self):
        self.loadDLTreeView()
        self.treeViewDLProgramEditor.clicked.connect(self.DLtreeViewClicked)
        self.pushButtonDLProgramExit.clicked.connect(self.goToScreenProgrammingMain)
        self.pushButtonDLProgramOKPath.clicked.connect(lambda: self.getpathName("OK"))
        self.pushButtonDLProgramNOKPath.clicked.connect(lambda: self.getpathName("NOK"))
        self.pushButtonDLProgramTrain.clicked.connect(self.trainButtonAction)
        self.pushButtonDLProgramPredict.clicked.connect(self.predictButtonAction)
        self.pushButtonDLProgramAddAugment.clicked.connect(self.addAugmentToDLTree)
        self.pushButtonDLProgramAddAugmentGroup.clicked.connect(self.addAugmentGroupToDLTreeView)
        self.pushButtonDLProgramRemoveAugment.clicked.connect(self.deleteAugmentFromDLTree)
        self.pushButtonDLProgramAugment.clicked.connect(self.augmentButtonAction)
        self.pushButtonDLProgramSave.clicked.connect(self.saveButtonDLProgramAction)

    def loadDLTreeView(self):
        #Configure QTreeView
        self.DLitemModel = QStandardItemModel()
        self.treeViewDLProgramEditor.setModel(self.DLitemModel)
        
    def launchNewDLProgram(self):
        self.DLitemModel.clear() #clear treeView itemModel
        self.DLmodel.clearProgram()
        self.labelImageScreenDLProgramEditor.clear()
        self.resetDLGeneralForm()
        parentItem = self.DLitemModel.invisibleRootItem()
        self.addInstructionToDLTreeView(DLPO.GROUP_NAME_STRING + "1", parentItem.text())
        self.DLmodel.addAugmentGroup()
        
    def resetDLGeneralForm(self):
        self.spinBoxDLProgramBatchSize.setValue(0)
        self.spinBoxDLProgramEpochs.setValue(0)
        self.spinBoxDLProgramTrainTestSplit.setValue(0)
        self.spinBoxDLProgramAugmentRuns.setValue(0)

    def addInstructionToDLTreeView(self, instructionName, parentName):
        if parentName == "":
            parentItem = self.DLitemModel.invisibleRootItem()
        else:
            parentItem = self.DLitemModel.findItems(parentName, flags = Qt.MatchRecursive)
            if len(parentItem) > 0:
                parentItem = parentItem[0]
        if parentItem:
            item = QStandardItem(instructionName)
            parentItem.appendRow(item)
            self.DLtreeIndex = item.index()
            self.treeViewDLProgramEditor.setCurrentIndex(self.DLtreeIndex)
            self.DLtreeViewClicked(item.index())

    def loadProgramToDLTreeViewScreenDLProgramEditor(self):
        self.DLitemModel.clear() #clear treeView itemModel
        self.resetDLGeneralForm()
        groups, augments, augmentsConfig = self.DLmodel.getProgramAttributes() #Get program
        for index, augment in enumerate(augments):
            self.addAugmentToDLTreeView(groups[index], augment)
            self.updateAugmentConfigurationDLTree(augmentsConfig[index])
            self.DLtreeViewClicked(self.DLtreeIndex)
        if len(augments) == 0:
            parentItem = self.DLitemModel.invisibleRootItem()
            self.addInstructionToDLTreeView(DLPO.GROUP_NAME_STRING + "1", parentItem.text())

    def addAugmentToDLTreeView(self, group, augment):
        #Check if group exists or add it to the treeView
        groupItem = self.DLitemModel.findItems(DLPO.GROUP_NAME_STRING + str(group))
        if not groupItem:
            parentItem = self.DLitemModel.invisibleRootItem()
            groupItem = QStandardItem(DLPO.GROUP_NAME_STRING + str(group))
            parentItem.appendRow(groupItem)
        else:
            groupItem = groupItem[0]
        item = QStandardItem(augment)
        groupItem.appendRow(item)
        self.DLtreeIndex = item.index()
        self.treeViewDLProgramEditor.setCurrentIndex(self.DLtreeIndex)
        self.DLtreeViewClicked(item.index())

    def DLtreeViewClicked(self, index):
        #Get configuration from currently selected tree command and pass it to vision program
        previousItem = self.DLitemModel.itemFromIndex(self.DLtreeIndex)
        configuration = self.getAugmentConfigurationFromDLTree()
        groupNumber = self.getAugmentGroupNumber(previousItem)
        if not previousItem.text().startswith(DLPO.GROUP_NAME_STRING):
            self.DLmodel.changeAugmentConfiguration(groupIndex = groupNumber, augmentName = previousItem.text(), configuration = configuration)
        self.DLtreeIndex = index #Update treeIndex
        #Update currently displayed widget according to type of instruction selected
        item = self.DLitemModel.itemFromIndex(index)
        groupNumber = self.getSelectedDLTreeViewGroupNumber()
        augmentType = item.text()
        self.updateAugmentVariableNames(augmentType)
        if not augmentType.startswith(DLPO.GROUP_NAME_STRING):
            augmentConfiguration = self.DLmodel.getAugmentConfiguration(groupIndex = groupNumber, augmentName = augmentType)
            self.updateAugmentConfigurationDLTree(augmentConfiguration)

    def getAugmentConfigurationFromDLTree(self):
        instructionData = []
        #stackCurrentWidgetName = self.stackedWidgetScreenProgramEditor.currentWidget().objectName()
        instructionData.append(self.spinBoxVariable1DLProgramEditor.value())
        instructionData.append(self.spinBoxVariable2DLProgramEditor.value())
        instructionData.append(self.spinBoxVariable3DLProgramEditor.value())
        instructionData.append(self.spinBoxVariable4DLProgramEditor.value())
        return instructionData

    def updateAugmentConfigurationDLTree(self, augmentConfig):
        self.spinBoxVariable1DLProgramEditor.setValue(augmentConfig[DLPO.AUGMENT_CONFIG_VARIABLES_1_KEY])
        self.spinBoxVariable2DLProgramEditor.setValue(augmentConfig[DLPO.AUGMENT_CONFIG_VARIABLES_2_KEY])
        self.spinBoxVariable3DLProgramEditor.setValue(augmentConfig[DLPO.AUGMENT_CONFIG_VARIABLES_3_KEY])
        self.spinBoxVariable4DLProgramEditor.setValue(augmentConfig[DLPO.AUGMENT_CONFIG_VARIABLES_4_KEY])

    def addAugmentToDLTree(self):
        #Launch dialog
        augmentSelectDialog = DialogAugmentSelection(self)
        augmentSelectDialog.exec()
        dialogReturnString = augmentSelectDialog.getReturnString()
        parentItem = self.DLitemModel.itemFromIndex(self.DLtreeIndex)
        if parentItem.parent() != None:
            parentItem = parentItem.parent()
        augmentType = dialogReturnString
        item = QStandardItem(augmentType)
        parentItem.appendRow(item)
        #Send information to DL model
        groupNumber = self.getSelectedDLTreeViewGroupNumber()
        self.DLmodel.addAugment(groupNumber, augmentType)
        #Select newly created index on treeView
        self.DLtreeViewClicked(item.index())
        self.treeViewDLProgramEditor.setCurrentIndex(item.index())

    def addAugmentGroupToDLTreeView(self):
        #Create new group and add it to the treeView
        parentItem = self.DLitemModel.invisibleRootItem()
        groupIndex = self.DLmodel.addAugmentGroup()
        item = QStandardItem(DLPO.GROUP_NAME_STRING + str(groupIndex))
        parentItem.appendRow(item)
        self.DLtreeIndex = item.index()
        self.DLtreeViewClicked(item.index())

    def getSelectedDLTreeViewGroupNumber(self):
        selectedIndex = QModelIndex(self.treeViewDLProgramEditor.selectedIndexes()[0])
        selectedItem = self.DLitemModel.itemFromIndex(selectedIndex)
        parentItem = selectedItem.parent()
        if parentItem != None:
            groupItem = parentItem
        else:
            groupItem = selectedItem
        groupName = groupItem.text()
        groupNumber = groupName[(groupName.rfind(" ")+1):]
        return groupNumber

    def getAugmentGroupNumber(self, item):
        parentItem = item.parent()
        if parentItem != None:
            groupItem = parentItem
        else:
            groupItem = item
        groupName = groupItem.text()
        groupNumber = groupName[(groupName.rfind(" ")+1):]
        return groupNumber

    def updateAugmentVariableNames(self, augmentType):
        variableNames = []
        labels = [self.labelVariable1DLProgramEditor, self.labelVariable2DLProgramEditor, self.labelVariable3DLProgramEditor, self.labelVariable4DLProgramEditor]
        spinBoxes = [self.spinBoxVariable1DLProgramEditor, self.spinBoxVariable2DLProgramEditor, self.spinBoxVariable3DLProgramEditor, self.spinBoxVariable4DLProgramEditor]
        variableNames = DLM.getAugmentVariableNames(augmentType)
        for index in range(len(labels)):
            labels[index].setText("N/A")
            if index < len(variableNames):
                labels[index].setText(variableNames[index])
                spinBoxes[index].setEnabled(True)
            else:
                spinBoxes[index].setEnabled(False)

    def deleteAugmentFromDLTree(self):
        selectedIndex = QModelIndex(self.treeViewDLProgramEditor.selectedIndexes()[0])
        selectedItem = self.DLitemModel.itemFromIndex(selectedIndex)
        groupNumber = self.getSelectedDLTreeViewGroupNumber()
        self.DLmodel.removeAugment(groupNumber, selectedItem.text())
        self.DLtreeIndex = selectedIndex.parent()
        self.DLitemModel.removeRow(selectedIndex.row(), selectedIndex.parent())
        item = self.DLitemModel.itemFromIndex(self.DLtreeIndex)
        counter = 0
        while (item == None):
            item = self.DLitemModel.itemFromIndex(self.DLitemModel.index(counter, 0))
            counter = counter + 1
            self.DLtreeIndex = self.DLitemModel.indexFromItem(item)
        self.DLtreeViewClicked(self.DLtreeIndex)
        self.treeViewDLProgramEditor.setCurrentIndex(self.DLtreeIndex)

    def getpathName(self, pathToSet):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(self,"Select Directory", options=options)
        if path:
            if pathToSet == "OK":
                self.DLmodel.setOKPath(path)
            elif pathToSet == "NOK":
                self.DLmodel.setNOKPath(path)

    def trainModelAndGetResult(self):
        batchSize = self.spinBoxDLProgramBatchSize.value()
        epochs = self.spinBoxDLProgramEpochs.value()
        trainTestSplit = self.spinBoxDLProgramTrainTestSplit.value()
        self.DLmodel.trainModel(epochs, trainTestSplit, batchSize)
        showImage = self.DLmodel.getTrainResultGraph()
        if len(showImage) > 0:
            self.setImageScreenDLProgramEditor(showImage)

    def trainButtonAction(self):
        #Launch dialog
        trainOutputDialog = DialogTrainOutput(self)
        trainOutputDialog.show()
        self.p = ProcessRunnable(target=self.trainModelAndGetResult, args=[])
        self.p.start()
        
    def predictButtonAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #modelFile, _ = QFileDialog.getOpenFileName(self,"Select Model", "","All Files (*);;Python Files (*.py)", options=options)
        imageFile, _ = QFileDialog.getOpenFileName(self,"Select Image", "","All Files (*);;Python Files (*.py)", options=options)
        prediction, showImage = self.DLmodel.modelPredict(imagePath = imageFile)
        self.setImageScreenDLProgramEditor(showImage)

    def augmentButtonAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(self,"Select Directory", options=options)
        augmentRuns = self.spinBoxDLProgramAugmentRuns.value()
        self.DLtreeViewClicked(self.DLtreeIndex)
        #Launch dialog
        augmentOutputDialog = DialogTrainOutput(self)
        augmentOutputDialog.show()
        self.p2 = ProcessRunnable(target=self.DLmodel.augmentImages, args=[path, augmentRuns])
        self.p2.start()
        #VER ACA SI CONVIENE HACER QUE EL OK PATH Y NOK PATH CAMBIEN AL DESTINATION PATH

    def saveButtonDLProgramAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path, filter = QFileDialog.getSaveFileName(self, "Save File", directory = PCP.PATH_SAVED_PROGRAMS, filter = "Program Files (*.json);;All Files (*)", options = options)
        self.DLmodel.saveProgram(path)

    def setImageScreenDLProgramEditor(self, image):
        try:
            tempImagePath = "temp/programImage.png"
            data = im.fromarray(image)
            data.save(tempImagePath)
            pixmap = QPixmap(tempImagePath)
            self.labelImageScreenDLProgramEditor.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")

    #End ScreenDLProgramEditor
    #########################################################


    #Class attributes
    itemModel = None
    DLitemModel = None
    treeIndex = None
    DLtreeIndex = None
    tableModel = None
    visionProgramStructure = ProgramStructure()
    DLmodel = DLM.modelDL()
    
    

#####################################################
#CLASS TABLE MODEL
#VER SI ESTA CLASE CORRESPONDE EN ESTE ARCHIVO
class MyTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data=[[]], parent=None):
        super().__init__(parent)
        self.data = data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == QtCore.Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headersorizontal[section]
            else:
                return "ID " + str(section+1)

    def columnCount(self, parent=None):
        if len(self.data)  > 0:
            return len(self.data[0])
        else:
            return 0

    def rowCount(self, parent=None):
        return len(self.data)

    def data(self, index: QModelIndex, role: int):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            return str(self.data[row][col])
    
    def getHeaders(self, headersorizontal, headersVertical):
        self.headersorizontal = headersorizontal
        #self.headersVertical = headersVertical #NO SE USA

    headersorizontal = None
    #headersVertical = None #NO SE USA

#END CLASS TABLE MODEL
#######################################################

#######################################################
#CLASS PROCESS RUNNABLE
class ProcessRunnable(QRunnable):
    def __init__(self, target, args):
        QRunnable.__init__(self)
        self.t = target
        self.args = args

    def run(self):
        self.t(*self.args)

    def start(self):
        QThreadPool.globalInstance().start(self)

#END CLASS PROCESS RUNNABLE
#######################################################

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())