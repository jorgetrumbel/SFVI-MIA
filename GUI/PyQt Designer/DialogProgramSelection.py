import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi
import VisionProgramOptions as VPO
import DeepLearningProgramOptions as DLPO

#############################################################
# DialogCommandSelection    
class DialogProgramSelection(QDialog):
    def __init__(self, parent=None):
        super(DialogProgramSelection, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogProgramSelection.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        self.loadTreeView()
        self.treeViewDialogProgramSelection.clicked.connect(self.treeViewClicked)

    def loadTreeView(self):
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        for itemName in VPO.visionProgramTypes:
            item = QStandardItem(itemName)
            parentItem.appendRow(item)
        #Create subitems for DL models
        parentItem = self.itemModel.item(VPO.visionProgramTypes.index(VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING))
        for itemName in DLPO.deepLearningModelNames:
            item = QStandardItem(itemName)
            parentItem.appendRow(item)

        self.treeViewDialogProgramSelection.setModel(self.itemModel)

    def treeViewClicked(self, index):
        self.treeIndex = index
        item = self.itemModel.itemFromIndex(index)
        self.dialogReturnString = item.text()

    def getReturnString(self):
        return self.dialogReturnString
    
    def getProgramType(self):
        returnString = None
        if VPO.VISION_PROGRAM_TYPES_CLASSIC == self.dialogReturnString:
            returnString = VPO.VISION_PROGRAM_TYPES_CLASSIC
        elif self.dialogReturnString in DLPO.deepLearningModelNames:
            returnString = VPO.VISION_PROGRAM_TYPES_DEEP_LEARNING
        return returnString

    itemModel = None
    treeIndex = None
    dialogReturnString = None
# End DialogCommandSelection
#############################################################