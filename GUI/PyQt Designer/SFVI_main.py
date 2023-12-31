import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFormLayout, QApplication, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem, QImage
from PyQt5.QtCore import QModelIndex, QAbstractTableModel, Qt, QRunnable, QThreadPool
from PyQt5.uic import loadUi
from PIL import Image as im

from DialogCommandSelection import DialogCommandSelection
from DialogProgramSelection import DialogProgramSelection
from DialogTrainOutput import DialogTrainOutput

import VisionProgramOptions as VPO
import DeepLearningProgramOptions as DLPO
from VisionProgram import ProgramStructure
import VisionProgram as VP
import DeepLearningModule as DLM
import DialogTrainOutput as DTO

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

    #########################################################
    #ScreenMonitorMain
    def ScreenMonitorMainLogic(self):
        self.buttonChangeToProgrammingMain.clicked.connect(self.goToScreenProgrammingMain)
        self.buttonSelectProgramScreenMonitorMain.clicked.connect(self.getProgramFileName)
        self.buttonCounterScreenMonitorMain.clicked.connect(self.triggerProgramRun) #CORREGIR - NO VA EN ESTE BOTON

    def goToScreenProgrammingMain(self):
        self.stackWidget.setCurrentWidget(self.ScreenProgrammingMain)

    def getProgramFileName(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self,"Select File", "","All Files (*);;Python Files (*.py)", options=options)
        if file:
            self.selectedProgram = file
            self.visionProgramStructure.loadProgram(file)
    
    def triggerProgramRun(self):
        image, data, dataType = self.visionProgramStructure.runProgram(True)
        #self.updateTableView(data, dataType) CORREGIR
        self.setImageScreenMonitorMain(image)
        
    def setImageScreenMonitorMain(self, image):
        try:
            tempImagePath = "temp/programImage.png"
            data = im.fromarray(image)
            data.save(tempImagePath)
            pixmap = QPixmap(tempImagePath)
            self.labelImageScreenMonitorMain.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")

    '''
    def setImageScreenMonitorMain(self):
        image = "images/apple.PNG"
        try:
            with open(image):
                pixmap = QPixmap(image)
                self.labelImageScreenMonitorMain.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")
    '''

    selectedProgram = None
    #End ScreenMonitorMain
    #########################################################

    #########################################################
    #ScreenProgrammingMain
    def ScreenProgrammingMainLogic(self):
        self.buttonChangeToMonitorMain.clicked.connect(self.goToScreenMonitorMain)
        self.buttonNewProgramScreenProgrammingMain.clicked.connect(self.launchNewProgramDialog)

    def goToScreenProgramEditor(self):
        self.stackWidget.setCurrentWidget(self.ScreenProgramEditor)

    def goToScreenMonitorMain(self):
        self.stackWidget.setCurrentWidget(self.ScreenMonitorMain)

    def goToScreenDLProgramEditor(self):
        self.stackWidget.setCurrentWidget(self.ScreenDLProgramEditor)

    def launchNewProgramDialog(self):
        #Launch dialog
        programSelectDialog = DialogProgramSelection(self)
        programSelectDialog.exec()
        programReturnString = programSelectDialog.getReturnString()
        programTypeString = programSelectDialog.getProgramType()
        if programTypeString == VPO.VISION_PROGRAM_TYPES_CLASSIC:
            self.goToScreenProgramEditor()
        elif programTypeString == VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING:
            self.DLmodel.setSelectedModel(programReturnString)
            self.goToScreenDLProgramEditor()

    #End ScreenProgrammingMain
    #########################################################

    #########################################################
    #ScreenProgramEditor
    def ScreenProgramEditorLogic(self):
        #self.setImageScreenProgramEditor()
        self.loadTreeView()
        self.treeViewScreenProgramEditor.clicked.connect(self.treeViewClicked)
        self.buttonExitScreenProgramEditor.clicked.connect(self.goToScreenProgrammingMain)
        self.buttonAddCommandScreenProgramEditor.clicked.connect(self.addCommandToTree)
        self.buttonRunScreenProgramEditor.clicked.connect(self.runVisionProgram)
        self.buttonDeleteCommandScreenProgramEditor.clicked.connect(self.deleteCommandFromTree)
        self.buttonFeatureDetectionTemplate.clicked.connect(lambda: self.visionProgramStructure.selectTemplate(self.getSelectedInstructionName(), self.getSelectedInstructionParentName()))
        self.buttonFilterCropArea.clicked.connect(lambda: self.visionProgramStructure.selectCropArea(self.getSelectedInstructionName(), self.getSelectedInstructionParentName()))
        self.buttonCaptureSelectFile.clicked.connect(self.getCaptureFileName)

    def setImageScreenProgramEditor(self, image):
        try:
            tempImagePath = "temp/programImage.png"
            data = im.fromarray(image)
            data.save(tempImagePath)
            pixmap = QPixmap(tempImagePath)
            self.labelImageScreenProgramEditor.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")

    def loadTreeView(self):
        #Setup first item for TreeView and configure QTreeView
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem("File Select1") #CORREGIR
        parentItem.appendRow(item)
        self.visionProgramStructure.addInstruction("File Select1", parentItem.text(), VPO.CAPTURE_OPTIONS_FILE_SELECT)
        self.treeIndex = item.index()
        self.treeViewScreenProgramEditor.setModel(self.itemModel)
        self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackCaptureOptions)
    
    def treeViewClicked(self, index):
        #Get configuration from currently selected tree command and pass it to vision program
        previousItem = self.itemModel.itemFromIndex(self.treeIndex)
        configuration = self.getInstructionConfigurationFromTree()
        self.visionProgramStructure.changeInstructionConfiguration(previousItem.text(), configuration)
        self.treeIndex = index #Update treeIndex
        #Update currently displayed widget according to type of instruction selected
        item = self.itemModel.itemFromIndex(index)
        instructionType = self.visionProgramStructure.getInstructionType(item.text())
        instructionConfiguration = self.visionProgramStructure.getInstructionConfiguration(item.text())
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
        self.spinBoxExposure.setValue(configuration[VPO.CAPTURE_CONFIGURATIONS_EXPOSURE])
    
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
        self.visionProgramStructure.addInstruction(dialogReturnString, parentItem.text(), instructionType)
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

    def getStackCaptureConfiguration(self):
        instructionData = {}
        instructionData[VPO.CAPTURE_CONFIGURATIONS_NAME] = self.lineEditCapturaName.text()
        instructionData[VPO.CAPTURE_CONFIGURATIONS_EXPOSURE] = self.spinBoxExposure.value()
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
        image, data, dataType = self.visionProgramStructure.runProgram(False, currentItem.text())
        self.updateTableView(data, dataType)
        self.setImageScreenProgramEditor(image)

    
    def getCaptureFileName(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self,"Select File", "","All Files (*);;Python Files (*.py)", options=options)
        if file:
            currentItem = self.itemModel.itemFromIndex(self.treeIndex)
            self.visionProgramStructure.getCaptureFileName(currentItem.text(), file)

    #End ScreenProgramEditor
    #########################################################

    #########################################################
    #ScreenDLProgramEditor
    def ScreenDLProgramEditorLogic(self):
        self.pushButtonDLProgramExit.clicked.connect(self.goToScreenProgrammingMain)
        self.pushButtonDLProgramOKPath.clicked.connect(lambda: self.getpathName("OK"))
        self.pushButtonDLProgramNOKPath.clicked.connect(lambda: self.getpathName("NOK"))
        self.pushButtonDLProgramTrain.clicked.connect(self.trainButtonAction)
        self.pushButtonDLProgramPredict.clicked.connect(self.predictButtonAction)

    def getpathName(self, pathToSet):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(self,"Select Directory", options=options)
        if path:
            if pathToSet == "OK":
                self.DLmodel.setOKPath(path)
            elif pathToSet == "NOK":
                self.DLmodel.setNOKPath(path)
    
    def trainButtonAction(self):
        #Launch dialog
        trainOutputDialog = DialogTrainOutput(self)
        trainOutputDialog.show()
        batchSize = self.spinBoxDLProgramBatchSize.value()
        epochs = self.spinBoxDLProgramEpochs.value()
        trainTestSplit = self.spinBoxDLProgramTrainTestSplit.value()
        self.p = ProcessRunnable(target=self.DLmodel.trainModel, args=[epochs, trainTestSplit])
        self.p.start()
        
    def predictButtonAction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #modelFile, _ = QFileDialog.getOpenFileName(self,"Select Model", "","All Files (*);;Python Files (*.py)", options=options)
        imageFile, _ = QFileDialog.getOpenFileName(self,"Select Image", "","All Files (*);;Python Files (*.py)", options=options)
        prediction, showImage = self.DLmodel.modelPredict(imagePath = imageFile)
        self.setImageScreenDLProgramEditor(showImage)

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
    treeIndex = None
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
        return len(self.data[0])

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