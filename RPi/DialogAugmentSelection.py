import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi
import DeepLearningProgramOptions as DLPO

#############################################################
# DialogCommandSelection    
class DialogAugmentSelection(QDialog):
    def __init__(self, parent=None):
        super(DialogAugmentSelection, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogAugmentSelection.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        self.loadTreeView()
        self.treeViewDialogAugmentSelection.clicked.connect(self.treeViewClicked)

    def loadTreeView(self):
        self.itemModel = QStandardItemModel()
        parentItem = self.itemModel.invisibleRootItem()
        for itemName in DLPO.augmentOptions:
            item = QStandardItem(itemName)
            parentItem.appendRow(item)

        self.treeViewDialogAugmentSelection.setModel(self.itemModel)

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