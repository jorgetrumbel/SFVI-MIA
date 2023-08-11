import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QDialog
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi

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
        loadUi("SFVI.ui", self)
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
        self.setImageScreenProgramEditor()
        self.loadTreeView()
        self.treeViewScreenProgramEditor.clicked.connect(self.treeViewClicked)
        self.buttonExitScreenProgramEditor.clicked.connect(self.goToScreenProgrammingMain)
        self.buttonAddCommandScreenProgramEditor.clicked.connect(self.addCommandToTree)

    def setImageScreenProgramEditor(self):
        image = "images/apple.PNG"
        try:
            with open(image):
                pixmap = QPixmap(image)
                self.labelImageScreenProgramEditor.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")

    def loadTreeView(self):
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem("Captura")
        parentItem.appendRow(item)
        parentItem = item
        item = QStandardItem("Filtro")
        parentItem.appendRow(item)
        self.treeViewScreenProgramEditor.setModel(self.itemModel)
    
    def treeViewClicked(self, index):
        self.treeIndex = index
        item = self.itemModel.itemFromIndex(index)
        if item.text() == "Captura":
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.stackCameraOptions)
        elif item.text() == "Filtro":
            self.stackedWidgetScreenProgramEditor.setCurrentWidget(self.StackFilterOptions)

    def addCommandToTree(self):
        #Launch dialog
        commandSelectDialog = DialogCommandSelection(self)
        commandSelectDialog.exec()
        print(commandSelectDialog.getReturnString())

        parentItem = self.itemModel.itemFromIndex(self.treeIndex)
        item = QStandardItem("Filtro2")
        parentItem.appendRow(item)


    itemModel = None
    treeIndex = None
    #End ScreenProgramEditor
    #########################################################
    
class DialogCommandSelection(QDialog):
    def __init__(self, parent=None):
        super(DialogCommandSelection, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("DialogCommandSelection.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        self.loadTreeView()
        self.treeViewDialogCommandSelection.clicked.connect(self.treeViewClicked)

    def loadTreeView(self):
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        #Create Capture row and subitems
        item = QStandardItem("Captura")
        parentItem.appendRow(item)
        parentItem = item
        item = QStandardItem("Camara")
        parentItem.appendRow(item)
        item = QStandardItem("Flash")
        parentItem.appendRow(item)
        #Create filter row and subitems
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem("Filtro")
        parentItem.appendRow(item)
        parentItem = item
        item = QStandardItem("Blur")
        parentItem.appendRow(item)
        item = QStandardItem("Gauss")
        parentItem.appendRow(item)
        item = QStandardItem("Sobel")
        parentItem.appendRow(item)
        #Create measure row and subitems
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem("Medición")
        parentItem.appendRow(item)
        parentItem = item
        item = QStandardItem("Regla")
        parentItem.appendRow(item)
        item = QStandardItem("Blobs")
        parentItem.appendRow(item)
        item = QStandardItem("Template matching")
        parentItem.appendRow(item)
        self.treeViewDialogCommandSelection.setModel(self.itemModel)

    def treeViewClicked(self, index):
        self.treeIndex = index
        item = self.itemModel.itemFromIndex(index)
        self.dialogReturnString = item.text()

    def getReturnString(self):
        return self.dialogReturnString
    
    itemModel = None
    treeIndex = None
    dialogReturnString = None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())