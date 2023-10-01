def getImplementedVisionFunctions():
    return filterOptions, captureOptions

instructionDataNames = ("Name", "Type", "Parent", "Configuration")
INSTRUCTION_DATA_NAME = instructionDataNames[0]
INSTRUCTION_DATA_TYPE = instructionDataNames[1]
INSTRUCTION_DATA_PARENT = instructionDataNames[2]
INSTRUCTION_DATA_CONFIGURATION = instructionDataNames[3]

commandGroups = ("Capture", "Filter", "Feature Detection")
COMMAND_GROUPS_CAPTURE = commandGroups[0]
COMMAND_GROUPS_FILTER = commandGroups[1]
COMMAND_GROUPS_FEATURE_DETECTION = commandGroups[2]

filterOptions = ("Blur", "Gauss", "Sobel", "Median", "Erosion", "Dilation", "Open", "Close",
                     "Gradient", "Top Hat", "Black Hat") #GUI names of filters, used for selection purposes
FILTER_OPTIONS_BLUR = filterOptions[0]
FILTER_OPTIONS_GAUSS = filterOptions[1]
FILTER_OPTIONS_SOBEL = filterOptions[2]
FILTER_OPTIONS_MEDIAN = filterOptions[3]
FILTER_OPTIONS_EROSION = filterOptions[4]
FILTER_OPTIONS_DILATION = filterOptions[5]
FILTER_OPTIONS_OPEN = filterOptions[6]
FILTER_OPTIONS_CLOSE = filterOptions[7]
FILTER_OPTIONS_GRADIENT = filterOptions[8]
FILTER_OPTIONS_TOPHAT = filterOptions[9]
FILTER_OPTIONS_BLACKHAT = filterOptions[10]

filterConfigurations = ("Name", "Kernel Rows", "Kernel Columns", "Iterations")
FILTER_CONFIGURATIONS_NAME = filterConfigurations[0]
FILTER_CONFIGURATIONS_KERNEL_ROWS = filterConfigurations[1]
FILTER_CONFIGURATIONS_KERNEL_COLUMNS = filterConfigurations[2]
FILTER_CONFIGURATIONS_ITERATIONS = filterConfigurations[3]

captureOptions = ("Camera", "Flash")
CAPTURE_OPTIONS_CAMERA = captureOptions[0]
CAPTURE_OPTIONS_FLASH = captureOptions[1]

captureConfigurations = ("Name", "Exposure")
CAPTURE_CONFIGURATIONS_NAME = captureConfigurations[0]
CAPTURE_CONFIGURATIONS_EXPOSURE = captureConfigurations[1]

featureDetectionOptions = ("Canny", "Hough")
FEATURE_DETECTION_OPTIONS_CANNY = featureDetectionOptions[0]
FEATURE_DETECTION_OPTIONS_HOUGH = featureDetectionOptions[1]

featureDetectionConfigurations = ("Name", "Variable 1", "Variable 2", "Variable 3")
FEATURE_DETECTION_CONFIGURATIONS_NAME = featureDetectionConfigurations[0]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1 = featureDetectionConfigurations[1]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2 = featureDetectionConfigurations[2]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3 = featureDetectionConfigurations[3]