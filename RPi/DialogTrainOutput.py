from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer
import LogModuleStdout as LOG


#############################################################
# DialogCommandSelection    
class DialogTrainOutput(QDialog):
    def __init__(self, parent=None):
        super(DialogTrainOutput, self).__init__(parent)
        LOG.buffer.seek(0)
        self.stdioCounter = len(LOG.buffer.read())
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogTrainOutput.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        print("Model Training Start")
        self.textEditTrainOutput.textChanged.connect(self.onUpdateText)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.textUpdateFunction)
        self.timer.start(200) # time in milliseconds.
        
    def textUpdateFunction(self):
        LOG.buffer.seek(self.stdioCounter)
        text = LOG.buffer.read()
        self.stdioCounter = self.stdioCounter + len(text)
        if text != '':
            self.textEditTrainOutput.append(text)

    def onUpdateText(self):
        cursor = self.textEditTrainOutput.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        #cursor.insertText(text)
        self.textEditTrainOutput.setTextCursor(cursor)
        self.textEditTrainOutput.ensureCursorVisible()
    

    itemModel = None
    treeIndex = None
    dialogReturnString = None
    stdioCounter = 0
    buffer = None
# End DialogCommandSelection
#############################################################