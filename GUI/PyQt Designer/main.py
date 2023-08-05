import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initializeUI()
        

    def initializeUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('SFVI - Sistema Flexible de Visión Industrial')
        self.setWindowIcon(QIcon('images/apple.PNG'))

        self.mainCentralWidget = QtWidgets.QWidget(self)
        self.mainCentralWidget.setEnabled(True)
        self.mainCentralWidget.setObjectName("mainCentalWidget")
        self.setCentralWidget(self.mainCentralWidget)
        self.maingridlayout = QtWidgets.QVBoxLayout(self.mainCentralWidget)
        self.maingridlayout.setObjectName("maingridLayout")
        self.mainCentralWidget.setFixedHeight(self.maingridlayout.sizeHint().height())
        #self.mainCentralWidget.setGeometry(0,0,1000,1000)
        self.stackWidget = QtWidgets.QStackedWidget(self.mainCentralWidget)

        self.setupScreens()

        self.show()

    def setupScreens(self):

        self.screenMonitorMain = ScreenMonitorMain()
        self.ScreenMonitorMainLogic()
        self.screenProgrammingMain = ScreenProgrammingMain()
        self.ScreenProgrammingMainLogic()
        self.stackWidget.addWidget(self.screenMonitorMain)
        self.stackWidget.addWidget(self.screenProgrammingMain)
        self.stackWidget.setGeometry(0, 0, 800, 600)

    def ScreenMonitorMainLogic(self):
        self.screenMonitorMain.buttonChangeToProgrammingMain.clicked.connect(self.goToScreenProgrammingMain)

    def ScreenProgrammingMainLogic(self):
        self.screenProgrammingMain.buttonChangeToMonitorMain.clicked.connect(self.goToScreenMonitorMain)

    def goToScreenProgrammingMain(self):
        self.stackWidget.setCurrentWidget(self.screenProgrammingMain)

    def goToScreenMonitorMain(self):
        self.stackWidget.setCurrentWidget(self.screenMonitorMain)

    stackWidget = None
    screenMonitorMain = None
    screenProgrammingMain = None

class ScreenMonitorMain(QMainWindow):
    def __init__(self):
        super(ScreenMonitorMain, self).__init__()
        loadUi("ScreenMonitoringMain.ui", self)
        
    
class ScreenProgrammingMain(QMainWindow):
    def __init__(self):
        super(ScreenProgrammingMain, self).__init__()
        loadUi("ScreenProgrammingMain.ui", self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())