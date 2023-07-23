import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from Registration import CreateNewUser # Import the registration module

class ButtonWindow(QWidget):
    def __init__(self):
        super().__init__() #Create default constructor for QWidget
        self.initializeUI()

    def initializeUI(self):
        #Initialize the window and display its contents to the screen
        self.setGeometry(100, 100, 400, 200)
        #self.setWindowTitle('QPushButton Widget')
        #self.displayButton() #Call displayButton function
        
        #self.setWindowTitle('QLineEdit Widget')
        #self.displayWidgets()

        #self.setWindowTitle('QCheckBox Widget')
        #self.displayCheckBoxes()

        #self.setWindowTitle('QMessageBox Example')
        #self.displayWidgets2()

        self.setWindowTitle('Login GUI')
        self.loginUserInterface()

        self.show()

    def displayButton(self):
        #Setup the button widget
        name_label = QLabel(self)
        name_label.setText("Dont push the button")
        name_label.move(60, 30) #Arrange the label
        button = QPushButton('Push Me', self)
        button.clicked.connect(self.buttonClicked)
        button.move(80, 70) #Arrange the button

    def buttonClicked(self):
        #Print message to the terminal and close the window when button is clicked
        print("The window has been closed")
        self.close()
    
    def displayWidgets(self):
        #Setup the QLineEdit and other widgets.
        # Create name label and line edit widgets
        QLabel("Please enter your name below.", self).move(100, 10)
        name_label = QLabel("Name:", self)
        name_label.move(70, 50)
        self.name_entry = QLineEdit(self)
        self.name_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.name_entry.move(130, 50)
        self.name_entry.resize(200, 20) # Change size of entry field
        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clearEntries)
        self.clear_button.move(160, 110)

    def clearEntries(self):
        #If button is pressed, clear the line edit input field.
        sender = self.sender()
        if sender.text() == 'Clear':
            self.name_entry.clear()


    def displayCheckBoxes(self):
        #Setup the checkboxes and other widgets
        header_label = QLabel(self)
        header_label.setText("Which shifts can you work? (Please check all that apply)")
        header_label.setWordWrap(True)
        header_label.move(10, 10)
        header_label.resize(230, 60)
        # Set up checkboxes
        morning_cb = QCheckBox("Morning [8 AM-2 PM]", self) # text, parent
        morning_cb.move(20, 80)
        #morning_cb.toggle() # uncomment if you want box to start off checked
        morning_cb.stateChanged.connect(self.printToTerminal)
        after_cb = QCheckBox("Afternoon [1 PM-8 PM]", self) # text, parent
        after_cb.move(20, 100)
        after_cb.stateChanged.connect(self.printToTerminal)
        night_cb = QCheckBox("Night [7 PM-3 AM]", self) # text, parent
        night_cb.move(20, 120)
        night_cb.stateChanged.connect(self.printToTerminal)

    def printToTerminal(self, state): # pass state of checkbox
        #Simple function to show how to determine the state of a checkbox. Prints the text label of the checkbox by determining which widget is sending the signal.
        sender = self.sender()
        if state == Qt.Checked:
            print("{} Selected.".format(sender.text()))
        else:
            print("{} Deselected.".format(sender.text()))

    def displayWidgets2(self):
        #Set up the widgets.
        catalogue_label = QLabel("Author Catalogue", self)
        catalogue_label.move(20, 20)
        catalogue_label.setFont(QFont('Arial', 20))
        auth_label = QLabel("Enter the name of the author you are searching for:", self)
        auth_label.move(40, 60)
        # Create author label and line edit widgets
        author_name = QLabel("Name:", self)
        author_name.move(50, 90)
        self.auth_entry = QLineEdit(self)
        self.auth_entry.move(95, 90)
        self.auth_entry.resize(240, 20)
        self.auth_entry.setPlaceholderText("firstname lastname")
        # Create search button
        search_button = QPushButton("Search", self)
        search_button.move(125, 130)
        search_button.resize(150, 40)
        search_button.clicked.connect(self.displayMessageBox)

    def displayMessageBox(self):
        #When button is clicked, search through catalogue of names. If name is found, display Author Found dialog. Otherwise, display Author Not Found dialog.
        # Check if authors.txt exists
        try:
            with open("files/authors.txt", "r") as f:
                # read each line into a list
                authors = [line.rstrip('\n') for line in f]
        except FileNotFoundError:
            print("The file cannot be found.")
        # Check for name in list
        not_found_msg = QMessageBox() # create not_found_msg object to avoid causing a 'referenced before assignment' error
        if self.auth_entry.text() in authors:
            QMessageBox().information(self, "Author Found", "Author found in catalogue!", QMessageBox.Ok, QMessageBox.Ok)
        else:
            not_found_msg = QMessageBox.question(self, "Author Not Found", "Author not found in catalogue.\nDo you wish to continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if not_found_msg == QMessageBox.No:
            print("Closing application.")
            self.close()
        else:
            pass


    def loginUserInterface(self):
        #Create the login GUI.
        login_label = QLabel(self)
        login_label.setText("login")
        login_label.move(180, 10)
        login_label.setFont(QFont('Arial', 20))
        # Username and password labels and line edit widgets
        name_label = QLabel("username:", self)
        name_label.move(30, 60)
        self.name_entry = QLineEdit(self)
        self.name_entry.move(110, 60)
        self.name_entry.resize(220, 20)
        password_label = QLabel("password:", self)
        password_label.move(30, 90)
        self.password_entry = QLineEdit(self)
        self.password_entry.move(110, 90)
        self.password_entry.resize(220, 20)
        # Sign in push button
        sign_in_button = QPushButton('login', self)
        sign_in_button.move(100, 140)
        sign_in_button.resize(200, 40)
        sign_in_button.clicked.connect(self.clickLogin)
        # Display show password checkbox
        show_pswd_cb = QCheckBox("show password", self)
        show_pswd_cb.move(110, 115)
        show_pswd_cb.stateChanged.connect(self.showPassword)
        show_pswd_cb.toggle()
        show_pswd_cb.setChecked(False)
        # Display sign up label and push button
        not_a_member = QLabel("not a member?", self)
        not_a_member.move(70, 200)
        sign_up = QPushButton("sign up", self)
        sign_up.move(160, 195)
        sign_up.clicked.connect(self.createNewUser)

    def clickLogin(self):
        #When user clicks sign in button, check if username and password match any existing profiles in users.txt.
        #If they exist, display messagebox and close program. If they don't, display error messagebox.
        users = {} # Create empty dictionary to store user information
        # Check if users.txt exists, otherwise create new file
        try:
            with open("files/users.txt", 'r') as f:
                for line in f:
                    user_fields = line.split(" ")
                    username = user_fields[0]
                    password = user_fields[1].strip('\n')
                    users[username] = password
        except FileNotFoundError:
            print("The file does not exist. Creating a new file.")
            f = open ("files/users.txt", "w")
        username = self.name_entry.text()
        password = self.password_entry.text()
        if (username, password) in users.items():
            QMessageBox.information(self, "Login Successful!", "Login Successful!", QMessageBox.Ok, QMessageBox.Ok)
            self.close() # close program
        else:
            QMessageBox.warning(self, "Error Message", "The username or password is incorrect.", QMessageBox.Close, QMessageBox.Close)

    def showPassword(self, state):
        #If checkbox is enabled, view password. Else, mask password so others cannot see it.
        if state == Qt.Checked:
            self.password_entry.setEchoMode(QLineEdit.Normal)
        else:
            self.password_entry.setEchoMode(QLineEdit.Password)

    def createNewUser(self):
        #When the sign up button is clicked, open a new window and allow the user to create a new account.
        self.create_new_user_dialog = CreateNewUser()
        self.create_new_user_dialog.show()


    def closeEvent(self, event):
        #Display a QMessageBox when asking the user if they want to quit the program.
        # Set up message box
        quit_msg = QMessageBox.question(self, "Quit Application?", "Are you sure you want to Quit?", QMessageBox.No | QMessageBox.Yes, QMessageBox.Yes)
        if quit_msg == QMessageBox.Yes:
            event.accept() # accept the event and close the application
        else:
            event.ignore() # ignore the close event

#Run program
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ButtonWindow()
    sys.exit(app.exec())
