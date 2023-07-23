import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QFormLayout, QLineEdit, QTextEdit, QSpinBox, QComboBox, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class SelectItems(QWidget):
    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        #Initialize the window and display its contents to the screen
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('ComboBox and SpinBox')
        #self.itemsAndPrices()
        self.formWidgets()
        self.show()

    def itemsAndPrices(self):
        #Create the widgets so users can select an item from the combo boxes
        #and a price from the spin boxes
        info_label = QLabel("Select 2 items you had for lunch and their prices.")
        info_label.setFont(QFont('Arial', 16))
        info_label.setAlignment(Qt.AlignCenter)
        self.display_total_label = QLabel("Total Spent: $")
        self.display_total_label.setFont(QFont('Arial', 16))
        self.display_total_label.setAlignment(Qt.AlignRight)
        # Create list of food items and add those items to two separate combo boxes
        lunch_list = ["egg", "turkey sandwich", "ham sandwich", "cheese",
        "hummus", "yogurt", "apple", "banana", "orange", "waffle", "baby carrots", 
        "bread", "pasta", "crackers", "pretzels", "pita chips",
        "coffee", "soda", "water"]
        lunch_cb1 = QComboBox()
        lunch_cb1.addItems(lunch_list)
        lunch_cb2 = QComboBox()
        lunch_cb2.addItems(lunch_list)
        # Create two separate price spin boxes
        self.price_sb1 = QSpinBox()
        self.price_sb1.setRange(0,100)
        self.price_sb1.setPrefix("$")
        self.price_sb1.valueChanged.connect(self.calculateTotal)
        self.price_sb2 = QSpinBox()
        self.price_sb2.setRange(0,100)
        self.price_sb2.setPrefix("$")
        self.price_sb2.valueChanged.connect(self.calculateTotal)
        # Create horizontal boxes to hold combo boxes and spin boxes
        h_box1 = QHBoxLayout()
        h_box2 = QHBoxLayout()
        h_box1.addWidget(lunch_cb1)
        h_box1.addWidget(self.price_sb1)
        h_box2.addWidget(lunch_cb2)
        h_box2.addWidget(self.price_sb2)
        # Add widgets and layouts to QVBoxLayout
        v_box = QVBoxLayout()
        v_box.addWidget(info_label)
        v_box.addLayout(h_box1)
        v_box.addLayout(h_box2)
        v_box.addWidget(self.display_total_label)
        self.setLayout(v_box)

    def calculateTotal(self):
        #Calculate and display total price from spin boxes and change value shown in QLabel
        total = self.price_sb1.value() + self.price_sb2.value()
        self.display_total_label.setText("Total Spent: ${}".format(str(total)))

    def formWidgets(self):
        #Create widgets that will be used in the application form.
        # Create widgets
        title = QLabel("Appointment Submission Form")
        title.setFont(QFont('Arial', 18))
        title.setAlignment(Qt.AlignCenter)
        name = QLineEdit()
        name.resize(100, 100)
        address = QLineEdit()
        mobile_num = QLineEdit()
        mobile_num.setInputMask("000-000-0000;")
        age_label = QLabel("Age")
        age = QSpinBox()
        age.setRange(1, 110)
        height_label = QLabel("Height")
        height = QLineEdit()
        height.setPlaceholderText("cm")
        weight_label = QLabel("Weight")
        weight = QLineEdit()
        weight.setPlaceholderText("kg")
        gender = QComboBox()
        gender.addItems(["Male", "Female"])
        surgery = QTextEdit()
        surgery.setPlaceholderText("separate by ','")
        blood_type = QComboBox()
        blood_type.addItems(["A", "B", "AB", "O"])
        hours = QSpinBox()
        hours.setRange(1, 12)
        minutes = QComboBox()
        minutes.addItems([":00", ":15", ":30", ":45"])
        am_pm = QComboBox()
        am_pm.addItems(["AM", "PM"])
        submit_button = QPushButton("Submit Appointment")
        submit_button.clicked.connect(self.close)
        # Create horizontal layout and add age, height, and weight to h_box
        h_box = QHBoxLayout()
        h_box.addSpacing(10)
        h_box.addWidget(age_label)
        h_box.addWidget(age)
        h_box.addWidget(height_label)
        h_box.addWidget(height)
        h_box.addWidget(weight_label)
        h_box.addWidget(weight)
        # Create horizontal layout and add time information
        desired_time_h_box = QHBoxLayout()
        desired_time_h_box.addSpacing(10)
        desired_time_h_box.addWidget(hours)
        desired_time_h_box.addWidget(minutes)
        desired_time_h_box.addWidget(am_pm)
        # Create form layout
        app_form_layout = QFormLayout()
        # Add all widgets to form layout
        app_form_layout.addRow(title)
        app_form_layout.addRow("Full Name", name)
        app_form_layout.addRow("Address", address)
        app_form_layout.addRow("Mobile Number", mobile_num)
        app_form_layout.addRow(h_box)
        app_form_layout.addRow("Gender", gender)
        app_form_layout.addRow("Past Surgeries ", surgery)
        app_form_layout.addRow("Blood Type", blood_type)
        app_form_layout.addRow("Desired Time", desired_time_h_box)
        app_form_layout.addRow(submit_button)
        self.setLayout(app_form_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SelectItems()
    sys.exit(app.exec_())