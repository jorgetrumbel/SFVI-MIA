
def getInstructionConfiguration(self, instructionType):
    #stackCurrentWidgetIndex = self.stackedWidgetScreenProgramEditor.currentIndex()
    instructionData = {}
    if instructionType in self.filterOptions:
        #stackCurrentWidget = self.stackedWidgetScreenProgramEditor.currentWidget()
        stackCurrentWidget = self.stackedWidgetScreenProgramEditor.widget(self.stackedWidgetScreenProgramEditor.indexOf(self.StackFilterOptions))
    elif instructionType in self.cameraOptions:
        stackCurrentWidget = self.stackedWidgetScreenProgramEditor.widget(self.stackedWidgetScreenProgramEditor.indexOf(self.stackCameraOptions))
    formLayout = stackCurrentWidget.findChildren(QFormLayout)
    formRows = formLayout[0].rowCount()
    for rowNumber in range(0,formRows):
        rowLabelText = formLayout[0].itemAt(rowNumber,0).widget().text()
        rowItemValue = formLayout[0].itemAt(rowNumber,1).widget().value()
        instructionData[rowLabelText] = rowItemValue
        #print(rowLabelText)
        #print(rowItemValue)
    #print(instructionData)
    return instructionData

def createProgramFromTree(self, inputItem, parentIndex, programData):
    rowItemNumber = inputItem.rowCount()
    for selectedRow in range(0,rowItemNumber):
        #Take the first item of the list
        currentRow = inputItem.child(selectedRow)
        currentIndex = parentIndex + selectedRow + 1
        instructionConfiguration = self.getInstructionConfiguration(currentRow.text())
        #print("parent:", parentIndex, currentRow.text(), "index:", currentIndex)
        programData[currentIndex] = {"Name": currentRow.text() + str(currentIndex),
                                                        "Type": currentRow.text(),
                                                        "Parent": parentIndex,
                                                        "Configuration": instructionConfiguration}
        if currentRow.hasChildren():
            #print(currentRow.rowCount())
            self.createProgramFromTree(currentRow, currentIndex, programData)
    if parentIndex == 0:
        #print(programData)
        with open("temp/program_file.json", "w") as write_file:
            json.dump(programData, write_file, indent=4)
    return programData

def runVisionProgram(self):
    program = self.createProgramFromTree(self.itemModel.invisibleRootItem(), 0, {})
    visionProgram = VisionProgram()
    visionProgram.loadImage("images/apple.png", grayscale=True)
    programLength = len(program) + 1
    for instructionNumber in range(1,programLength):
        instruction = program[instructionNumber]
        instructionConfiguration = instruction["Configuration"]
        if instruction["Type"] == "Blur":
            visionProgram.applyBlurFilter(instructionConfiguration["Kernel rows"], instructionConfiguration["Kernel Columns"])
        elif instruction["Type"] == "Gauss":
            visionProgram.applyGaussFilter(instructionConfiguration["Kernel rows"], instructionConfiguration["Kernel Columns"])
        elif instruction["Type"] == "Sobel":
            visionProgram.applySobelFilter()
    image = visionProgram.getImage()
    #visionProgram.showImage()
    self.setImageScreenProgramEditor(image)

def getInstructionConfigurationFromTree(self):
    instructionData = {}
    stackCurrentWidget = self.stackedWidgetScreenProgramEditor.currentWidget()
    formLayout = stackCurrentWidget.findChildren(QFormLayout)
    formRows = formLayout[0].rowCount()
    for rowNumber in range(0,formRows):
        rowLabelText = formLayout[0].itemAt(rowNumber,0).widget().text()
        rowItemValue = formLayout[0].itemAt(rowNumber,1).widget().value()
        instructionData[rowLabelText] = rowItemValue
    return instructionData