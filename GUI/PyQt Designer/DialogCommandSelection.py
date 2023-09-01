import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi
import VisionProgram

#############################################################
# DialogCommandSelection    
class DialogCommandSelection(QDialog):
    def __init__(self, parent=None):
        super(DialogCommandSelection, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogCommandSelection.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        self.loadTreeView()
        self.treeViewDialogCommandSelection.clicked.connect(self.treeViewClicked)

    def loadTreeView(self):
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        #Create Capture row and subitems
        item = QStandardItem(VisionProgram.COMMAND_GROUPS_CAPTURE)
        parentItem.appendRow(item)
        parentItem = item
        for itemName in VisionProgram.captureOptions:
            item = QStandardItem(itemName)
            parentItem.appendRow(item)
        #Create filter row and subitems
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem(VisionProgram.COMMAND_GROUPS_FILTER)
        parentItem.appendRow(item)
        parentItem = item
        for itemName in VisionProgram.filterOptions:
            item = QStandardItem(itemName)
            parentItem.appendRow(item)
        #Create feature detection row and subitems
        parentItem = self.itemModel.invisibleRootItem()
        item = QStandardItem(VisionProgram.COMMAND_GROUPS_FEATURE_DETECTION)
        parentItem.appendRow(item)
        parentItem = item
        for itemName in VisionProgram.featureDetectionOptions:
            item = QStandardItem(itemName)
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
# End DialogCommandSelection
#############################################################