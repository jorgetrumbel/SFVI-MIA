import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFormLayout, QApplication, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QDialog
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem, QImage
from PyQt5.QtCore import QModelIndex
from PyQt5.uic import loadUi
from PIL import Image as im
import json

from DialogCommandSelection import DialogCommandSelection
from VisionProgram import VisionProgram

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

    #########################################################
    #ScreenMonitorMain
    def ScreenMonitorMainLogic(self):
        self.buttonChangeToProgrammingMain.clicked.connect(self.goToScreenProgrammingMain)
        self.setImageScreenMonitorMain()

    def goToScreenProgrammingMain(self):
        self.stackWidget.setCurrentWidget(self.ScreenProgrammingMain)
    #End ScreenMonitorMain
    #########################################################

    #########################################################
    #ScreenProgrammingMain
    def ScreenProgrammingMainLogic(self):
        self.buttonChangeToMonitorMain.clicked.connect(self.goToScreenMonitorMain)
        self.buttonNewProgramScreenProgrammingMain.clicked.connect(self.goToScreenProgramEditor)

    def goToScreenProgramEditor(self):
        self.stackWidget.setCurrentWidget(self.ScreenProgramEditor)

    def goToScreenMonitorMain(self):
        self.stackWidget.setCurrentWidget(self.ScreenMonitorMain)

    def setImageScreenMonitorMain(self):
        image = "images/apple.PNG"
        try:
            with open(image):
                pixmap = QPixmap(image)
                self.labelImageScreenMonitorMain.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")
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
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem("Captura")
        parentItem.appendRow(item)
        '''
        parentItem = item
        item = QStandardItem("Filtro")
        parentItem.appendRow(item)
        '''
        self.treeViewScreenProgramEditor.setModel(self.itemModel)
    
    def treeViewClicked(self, index):
        self.treeIndex = index
        
        item = self.itemModel.itemFromIndex(index)
        if item.text() == "Captura":
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackCameraOptions)
        elif item.text() in self.filterOptions:
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.StackFilterOptions)

    def addCommandToTree(self):
        #Launch dialog
        commandSelectDialog = DialogCommandSelection(self)
        commandSelectDialog.exec()
        dialogReturnString = commandSelectDialog.getReturnString()
        parentItem = self.itemModel.itemFromIndex(self.treeIndex)
        item = QStandardItem(dialogReturnString)
        parentItem.appendRow(item)

    def deleteCommandFromTree(self):
        selectedIndex = QModelIndex(self.treeViewScreenProgramEditor.selectedIndexes()[0])
        self.itemModel.removeRow(selectedIndex.row(), selectedIndex.parent())

    def createProgramFromTree(self, inputItem, parentIndex, programData):
        rowItemNumber = inputItem.rowCount()
        for selectedRow in range(0,rowItemNumber):
            #Take the first item of the list
            currentRow = inputItem.child(selectedRow)
            currentIndex = parentIndex + selectedRow + 1
            instructionConfiguration = self.getInstructionConfiguration(currentRow.text())
            #print("parent:", parentIndex, currentRow.text(), "index:", currentIndex)
            programData[currentIndex] = {"Name": currentRow.text() + str(currentIndex),
                                                          "Type": currentRow.text(),
                                                          "Parent": parentIndex,
                                                          "Configuration": instructionConfiguration}
            if currentRow.hasChildren():
                #print(currentRow.rowCount())
                self.createProgramFromTree(currentRow, currentIndex, programData)
        if parentIndex == 0:
            #print(programData)
            with open("temp/program_file.json", "w") as write_file:
                json.dump(programData, write_file, indent=4)
        return programData
            
    def getInstructionConfiguration(self, instructionType):
        #stackCurrentWidgetIndex = self.stackedWidgetScreenProgramEditor.currentIndex()
        instructionData = {}
        if instructionType in self.filterOptions:
            #stackCurrentWidget = self.stackedWidgetScreenProgramEditor.currentWidget()
            stackCurrentWidget = self.stackedWidgetScreenProgramEditor.widget(self.stackedWidgetScreenProgramEditor.indexOf(self.StackFilterOptions))
        elif instructionType in self.cameraOptions:
            stackCurrentWidget = self.stackedWidgetScreenProgramEditor.widget(self.stackedWidgetScreenProgramEditor.indexOf(self.stackCameraOptions))
        formLayout = stackCurrentWidget.findChildren(QFormLayout)
        formRows = formLayout[0].rowCount()
        for rowNumber in range(0,formRows):
            rowLabelText = formLayout[0].itemAt(rowNumber,0).widget().text()
            rowItemValue = formLayout[0].itemAt(rowNumber,1).widget().value()
            instructionData[rowLabelText] = rowItemValue
            #print(rowLabelText)
            #print(rowItemValue)
        #print(instructionData)
        return instructionData
            

    def runVisionProgram(self):
        program = self.createProgramFromTree(self.itemModel.invisibleRootItem(), 0, {})
        visionProgram = VisionProgram()
        visionProgram.loadImage("images/apple.png", grayscale=True)
        programLength = len(program) + 1
        for instructionNumber in range(1,programLength):
            instruction = program[instructionNumber]
            instructionConfiguration = instruction["Configuration"]
            if instruction["Type"] == "Blur":
                visionProgram.applyBlurFilter(instructionConfiguration["Kernel rows"], instructionConfiguration["Kernel Columns"])
            elif instruction["Type"] == "Gauss":
                visionProgram.applyGaussFilter(instructionConfiguration["Kernel rows"], instructionConfiguration["Kernel Columns"])
            elif instruction["Type"] == "Sobel":
                visionProgram.applySobelFilter()
        image = visionProgram.getImage()
        #visionProgram.showImage()
        self.setImageScreenProgramEditor(image)

    itemModel = None
    treeIndex = None
    filterOptions = ("Filtro", "Blur", "Gauss", "Sobel") #GUI names of filters, used for selection purposes
    cameraOptions = ("Camara", "Captura", "Flash")
    #End ScreenProgramEditor
    #########################################################
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())