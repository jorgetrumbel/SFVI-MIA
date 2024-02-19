import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi
import ProgramConfigOptions as PCO

#############################################################
# DialogGeneralConfig   
class DialogGeneralConfig(QDialog):
    def __init__(self, parent=None):
        super(DialogGeneralConfig, self).__init__(parent)
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 500)
        self.setWindowIcon(QIcon('images/apple.PNG'))
        loadUi("ui/DialogGeneralConfiguration.ui", self)
        self.setupLogic()
        self.show()

    def setupLogic(self):
        self.pushButtonProgramOnline.setStyleSheet('QPushButton {background-color: red;}')
        self.pushButtonProgramOnline.setText(PCO.PROGRAM_ONLINE_STATUS_OFFLINE)
        self.buttonOk.clicked.connect(self.close)
        self.pushButtonProgramOnline.clicked.connect(self.programStatusButtonAction)
        self.pushButtonResetCounters.clicked.connect(self.resetButtonAction)

    def programStatusButtonAction(self):
        if self.pushButtonProgramOnline.isChecked():
            self.pushButtonProgramOnline.setStyleSheet('QPushButton {background-color: green;}')
            self.pushButtonProgramOnline.setText(PCO.PROGRAM_ONLINE_STATUS_ONLINE)
        else:
            self.pushButtonProgramOnline.setStyleSheet('QPushButton {background-color: red;}')
            self.pushButtonProgramOnline.setText(PCO.PROGRAM_ONLINE_STATUS_OFFLINE)
            
    def resetButtonAction(self): 
        self.resetButtonPushed = True

    def checkIfResetWasPressed(self):
        return self.resetButtonPushed

    def getProgramStatus(self):
        returnStatus = PCO.PROGRAM_ONLINE_STATUS_OFFLINE
        if self.pushButtonProgramOnline.isChecked():
            returnStatus = PCO.PROGRAM_ONLINE_STATUS_ONLINE
        return returnStatus
    
    resetButtonPushed = False

# End DialogCommandSelection
#############################################################