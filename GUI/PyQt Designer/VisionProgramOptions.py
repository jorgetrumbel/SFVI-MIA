def getImplementedVisionFunctions():
    return filterOptions, captureOptions

instructionDataNames = ("Name", "Type", "Parent", "Configuration")
INSTRUCTION_DATA_NAME = instructionDataNames[0]
INSTRUCTION_DATA_TYPE = instructionDataNames[1]
INSTRUCTION_DATA_PARENT = instructionDataNames[2]
INSTRUCTION_DATA_CONFIGURATION = instructionDataNames[3]

commandGroups = ("Capture", "Filter", "Feature Detection", "Draw")
COMMAND_GROUPS_CAPTURE = commandGroups[0]
COMMAND_GROUPS_FILTER = commandGroups[1]
COMMAND_GROUPS_FEATURE_DETECTION = commandGroups[2]
COMMAND_GROUPS_DRAW = commandGroups[3]

#FILTER OPTIONS
filterOptions = ("Blur", "Gauss", "Sobel", "Median", "Erosion", "Dilation", "Open", "Close",
                     "Gradient", "Top Hat", "Black Hat", "Histogram", "Threshold", "Range Threshold",
                     "Otsu Threshold", "Adaptative Gaussian Threshold", "Canny", "Auto-Canny") #GUI names of filters, used for selection purposes
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
FILTER_OPTIONS_HISTOGRAM = filterOptions[11]
FILTER_OPTIONS_THRESHOLD = filterOptions[12]
FILTER_OPTIONS_THRESHOLD_RANGE = filterOptions[13]
FILTER_OPTIONS_THRESHOLD_OTSU = filterOptions[14]
FILTER_OPTIONS_THRESHOLD_ADAPTATIVE_GAUSSIAN = filterOptions[15]
FILTER_OPTIONS_CANNY = filterOptions[16]
FILTER_OPTIONS_CANNY_AUTO = filterOptions[17]


filterConfigurations = ("Name", "Kernel Rows", "Kernel Columns", "Iterations", "Threshold", "Threshold 2")
FILTER_CONFIGURATIONS_NAME = filterConfigurations[0]
FILTER_CONFIGURATIONS_KERNEL_ROWS = filterConfigurations[1]
FILTER_CONFIGURATIONS_KERNEL_COLUMNS = filterConfigurations[2]
FILTER_CONFIGURATIONS_ITERATIONS = filterConfigurations[3]
FILTER_CONFIGURATIONS_THRESHOLD = filterConfigurations[4]
FILTER_CONFIGURATIONS_THRESHOLD2 = filterConfigurations[5]

#CAPTURE OPTIONS
captureOptions = ("Camera", "Flash")
CAPTURE_OPTIONS_CAMERA = captureOptions[0]
CAPTURE_OPTIONS_FLASH = captureOptions[1]

captureConfigurations = ("Name", "Exposure")
CAPTURE_CONFIGURATIONS_NAME = captureConfigurations[0]
CAPTURE_CONFIGURATIONS_EXPOSURE = captureConfigurations[1]

#FEATURE DETECTION OPTIONS
featureDetectionOptions = ("Contours", "Hough", "Probabilistic Hough", "Line Detector", "Template Matching",
                           "Template Matching Multiple", "Template Matching Invariant", "Canny Template Match",
                           "Canny Template Match Invariant")
FEATURE_DETECTION_OPTIONS_CONTOURS = featureDetectionOptions[0]
FEATURE_DETECTION_OPTIONS_HOUGH = featureDetectionOptions[1]
FEATURE_DETECTION_OPTIONS_HOUGH_PROBABILISTIC = featureDetectionOptions[2]
FEATURE_DETECTION_OPTIONS_LINE_DETECTOR = featureDetectionOptions[3]
FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH = featureDetectionOptions[4]
FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_MULTIPLE = featureDetectionOptions[5]
FEATURE_DETECTION_OPTIONS_TEMPLATE_MATCH_INVARIANT = featureDetectionOptions[6]
FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH = featureDetectionOptions[7]
FEATURE_DETECTION_OPTIONS_CANNY_TEMPLATE_MATCH_INVARIANT = featureDetectionOptions[8]

featureDetectionConfigurations = ("Name", "Variable 1", "Variable 2", "Variable 3")
FEATURE_DETECTION_CONFIGURATIONS_NAME = featureDetectionConfigurations[0]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_1 = featureDetectionConfigurations[1]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_2 = featureDetectionConfigurations[2]
FEATURE_DETECTION_CONFIGURATIONS_VARIABLE_3 = featureDetectionConfigurations[3]

#DRAW OPTIONS
drawOptions = ("Bounding Boxes", "Min Area Rectangles", "Draw Canny", "Draw Auto Canny", "Point Distance",
               "Segment Min Distance", "Detected Hough Lines", "Detected Probabilistic Hough Lines",
               "Segment Detector Lines", "Draw Template Match", "Draw Multiple Template Match", "Draw Template Match Invariant")
DRAW_OPTIONS_BOUNDING_BOXES = drawOptions[0]
DRAW_OPTIONS_MIN_AREA_RECTANGLES = drawOptions[1]
DRAW_OPTIONS_CANNY = drawOptions[2]
DRAW_OPTIONS_AUTO_CANNY = drawOptions[3]
DRAW_OPTIONS_POINT_DISTANCE = drawOptions[4]
DRAW_OPTIONS_SEGMENT_MIN_DISTANCE = drawOptions[5]
DRAW_OPTIONS_DETECTED_HOUGH_LINES = drawOptions[6]
DRAW_OPTIONS_DETECTED_PROBABILISTIC_HOUGH_LINES = drawOptions[7]
DRAW_OPTIONS_SEGMENT_DETECTOR_LINES = drawOptions[8]
DRAW_OPTIONS_TEMPLATE_MATCH = drawOptions[9]
DRAW_OPTIONS_MULTIPLE_TEMPLATE_MATCH = drawOptions[10]
DRAW_OPTIONS_TEMPLATE_MATCH_INVARIANT = drawOptions[11]

drawOptionsConfigurations = ("Name", "Variable 1", "Variable 2", "Variable 3")
DRAW_OPTIONS_CONFIGURATIONS_NAME = drawOptionsConfigurations[0]
DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_1 = drawOptionsConfigurations[1]
DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_2 = drawOptionsConfigurations[2]
DRAW_OPTIONS_CONFIGURATIONS_VARIABLE_3 = drawOptionsConfigurations[3]