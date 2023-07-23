import sys, os.path
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QFont

class EmptyWindow(QWidget):
    def __init__(self):
        super().__init__() #Create default constructor for QWidget
        self.initializeUI()

    def initializeUI(self):
        #Initialize the window and display its contents to the screen.
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle('Empty Window in PyQt')
        self.show()

class HelloWorldWindow(QWidget):
    def __init__(self):
        super().__init__() #Create default constructor for QWidget
        self.initializeUI()

    def initializeUI(self):
        #Initialize the window and display its contents to the screen.
        self.setGeometry(100, 100, 250, 250)
        self.setWindowTitle('QLabel Example')
        self.displayLabels()
        self.show()

    def displayLabels(self):
        #Display text and images using QLabels. Check to see if image files exist, if not throw an exception.
        text = QLabel(self)
        text.setText("Hello")
        text.move(105, 15)
        image = "images/world.PNG"
        try:
            with open(image):
                world_image = QLabel(self)
                pixmap = QPixmap(image)
                world_image.setPixmap(pixmap)
                world_image.move(25, 40)
        except FileNotFoundError:
            print("Image not found.")

class UserProfile(QWidget):
    def __init__(self):
        super().__init__()
        self.initializeUI()
    
    def initializeUI(self):
        #Initialize the window and display its contents to the screen.
        self.setGeometry(50, 50, 250, 400)
        self.setWindowTitle("User Profile GUI")
        self.displayImages()
        self.displayUserInfo()
        self.show()

    def displayImages(self):
        #Display background and profile images. Check to see if image files exist, if not throw an exception.
        background_image = "images/skublue.PNG"
        profile_image = "images/profile_image.PNG"
        try:
            with open(background_image):
                background = QLabel(self)
                pixmap = QPixmap(background_image)
                background.setPixmap(pixmap)
        except FileNotFoundError:
            print("Image not found.")
        try:
            with open(profile_image):
                user_image = QLabel(self)
                pixmap = QPixmap(profile_image)
                user_image.setPixmap(pixmap)
                user_image.move(80, 20)
        except FileNotFoundError:
            print("Image not found.")

    def displayUserInfo(self):
        #Create the labels to be displayed for the User Profile.
        user_name = QLabel(self)
        user_name.setText("John Doe")
        user_name.move(85, 140)
        user_name.setFont(QFont('Arial', 20))
        bio_title = QLabel(self)
        bio_title.setText("Biography")
        bio_title.move(15, 170)
        bio_title.setFont(QFont('Arial', 17))
        about = QLabel(self)
        about.setText("I'm a Software Engineer with 8 years experience creating awesome code.")
        about.setWordWrap(True)
        about.move(15, 190)
        skills_title = QLabel(self)
        skills_title.setText("Skills")
        skills_title.move(15, 240)
        skills_title.setFont(QFont('Arial', 17))
        skills = QLabel(self)
        skills.setText("Python | PHP | SQL | JavaScript")
        skills.move(15, 260)
        experience_title = QLabel(self)
        experience_title.setText("Experience")
        experience_title.move(15, 290)
        experience_title.setFont(QFont('Arial', 17))
        experience = QLabel(self)
        experience.setText("Python Developer")
        experience.move(15, 310)
        dates = QLabel(self)
        dates.setText("Mar 2011 - Present")
        dates.move(15, 330)
        dates.setFont(QFont('Arial', 10))
        experience = QLabel(self)
        experience.setText("Pizza Delivery Driver")
        experience.move(15, 350)
        dates = QLabel(self)
        dates.setText("Aug 2015 - Dec 2017")
        dates.move(15, 370)
        dates.setFont(QFont('Arial', 10))

# Run program
if __name__ == '__main__':
    app = QApplication(sys.argv) #Create application object
    window = UserProfile() #Create window
    sys.exit(app.exec_()) # Start the event loop and use sys.exit to close the application