deepLearningModelNames = ("VGG16", "Resnet18", "EfficientNet", "RegNet128", "SWIN", "YOLOv8")
DL_MODEL_NAME_VGG16 = deepLearningModelNames[0]
DL_MODEL_NAME_RESNET18 = deepLearningModelNames[1]
DL_MODEL_NAME_EFFICIENTNET = deepLearningModelNames[2]
DL_MODEL_NAME_REGNET128 = deepLearningModelNames[3]
DL_MODEL_NAME_SWIN = deepLearningModelNames[4]
DL_MODEL_NAME_YOLOV8 = deepLearningModelNames[5]

#GROUP NAMING
GROUP_NAME_STRING = "Augment Group "

#CLASSES
classes = ["OK", "NOK"]
DL_CLASS_OK = classes[0]
DL_CLASS_NOK = classes[1]

CLASS_QUANTITY = len(classes)

#GENERAL TRAIN CONFIG
MODEL_TRANSFORM_IMAGE_WIDTH = 150
MODEL_TRANSFORM_IMAGE_HEIGHT = 150

#PATHS
#PATH_TEMP_AUGMENTATION = "C:\\Users\\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\imageAugmentation"
PATH_TEMP_MODEL_SAVE = "C:\\Users\\Alejandro\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\Models"
PATH_TEMP_MODEL_RESULTS = "C:\\Users\Alejandro\\Desktop\\MIA Trabajo final\\Repositorio\\SFVI-MIA\\GUI\\PyQt Designer\\temp\\DeepLearning\\Results"

#AUGMENT OPTIONS
augmentOptions = ("Resize", "Random Resize", "Random Crop", "Center Crop", 
                  "Random Horizontal Flip", "Random Vertical Flip", "Pad", "Random Rotation",
                  "Random Affine", "Random Perspective", "Color Jitter",
                  "Gaussian Blur", "Random Invert") #GUI names of augments, used for selection purposes
AUGMENT_OPTIONS_RESIZE = augmentOptions[0]
AUGMENT_OPTIONS_RANDOM_RESIZE = augmentOptions[1]
AUGMENT_OPTIONS_RANDOM_CROP = augmentOptions[2]
AUGMENT_OPTIONS_CENTER_CROP = augmentOptions[3]
AUGMENT_OPTIONS_RANDOM_HORIZONTAL_FLIP = augmentOptions[4]
AUGMENT_OPTIONS_RANDOM_VERTICAL_FLIP = augmentOptions[5]
AUGMENT_OPTIONS_PAD = augmentOptions[6]
AUGMENT_OPTIONS_RANDOM_ROTATION = augmentOptions[7]
AUGMENT_OPTIONS_RANDOM_AFFINE = augmentOptions[8]
AUGMENT_OPTIONS_RANDOM_PERSPECTIVE = augmentOptions[9]
AUGMENT_OPTIONS_COLOR_JITTER = augmentOptions[10]
AUGMENT_OPTIONS_GAUSSIAN_BLUR = augmentOptions[11]
AUGMENT_OPTIONS_RANDOM_INVERT = augmentOptions[12]

augmentConfigVariables = (1,2,3,4)
AUGMENT_CONFIG_VARIABLES_1 = augmentConfigVariables[0]
AUGMENT_CONFIG_VARIABLES_2 = augmentConfigVariables[1]
AUGMENT_CONFIG_VARIABLES_3 = augmentConfigVariables[2]
AUGMENT_CONFIG_VARIABLES_4 = augmentConfigVariables[3]

#RESIZE AUGMENT CONFIG OPTIONS
AUGMENT_RESIZE_CONFIG_SIZE_H = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_RESIZE_CONFIG_SIZE_W = AUGMENT_CONFIG_VARIABLES_2

#RANDOM RESIZE AUGMENT CONFIG OPTIONS
AUGMENT_RANDOM_RESIZE_CONFIG_SIZE_MIN = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_RANDOM_RESIZE_CONFIG_SIZE_MAX = AUGMENT_CONFIG_VARIABLES_2

#RANDOM CROP AUGMENT CONFIG OPTIONS
AUGMENT_RANDOM_CROP_CONFIG_SIZE_H = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_RANDOM_CROP_CONFIG_SIZE_W = AUGMENT_CONFIG_VARIABLES_2
AUGMENT_RANDOM_CROP_CONFIG_PADDING = AUGMENT_CONFIG_VARIABLES_3

#CENTER CROP AUGMENT CONFIG OPTIONS
AUGMENT_CENTER_CROP_CONFIG_SIZE_H = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_CENTER_CROP_CONFIG_SIZE_W = AUGMENT_CONFIG_VARIABLES_2

#RANDOM HORIZONTAL FLIP CONFIG OPTIONS
AUGMENT_RANDOM_HORIZONTAL_FLIP_CONFIG_PROB = AUGMENT_CONFIG_VARIABLES_1

#RANDOM VERTICAL FLIP CONFIG OPTIONS
AUGMENT_RANDOM_VERTICAL_FLIP_CONFIG_PROB = AUGMENT_CONFIG_VARIABLES_1

#PAD CONFIG OPTIONS
AUGMENT_PAD_CONFIG_PADDING = AUGMENT_CONFIG_VARIABLES_1

#RANDOM ROTATION CONFIG OPTIONS
AUGMENT_RANDOM_ROTATION_CONFIG_DEGREES = AUGMENT_CONFIG_VARIABLES_1

#RANDOM AFFINE OPTIONS
AUGMENT_RANDOM_AFFINE_CONFIG_DEGREES = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_RANDOM_AFFINE_CONFIG_TRANSLATE = AUGMENT_CONFIG_VARIABLES_2
AUGMENT_RANDOM_AFFINE_CONFIG_SCALE = AUGMENT_CONFIG_VARIABLES_3
AUGMENT_RANDOM_AFFINE_CONFIG_SHEAR = AUGMENT_CONFIG_VARIABLES_4

#RANDOM PERSPECTIVE OPTIONS
AUGMENT_RANDOM_PERSPECTIVE_CONFIG_DISTORTION_PERCENT = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_RANDOM_PERSPECTIVE_CONFIG_PROBABILITY = AUGMENT_CONFIG_VARIABLES_2

#COLOR JITTER OPTIONS
AUGMENT_COLOR_JITTER_CONFIG_BRIGHTNESS = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_COLOR_JITTER_CONFIG_CONTRAST = AUGMENT_CONFIG_VARIABLES_2
AUGMENT_COLOR_JITTER_CONFIG_SATURATION = AUGMENT_CONFIG_VARIABLES_3
AUGMENT_COLOR_JITTER_CONFIG_HUE = AUGMENT_CONFIG_VARIABLES_4

#GAUSSIAN BLUR OPTIONS
AUGMENT_GAUSSIAN_BLUR_CONFIG_KERNEL = AUGMENT_CONFIG_VARIABLES_1
AUGMENT_GAUSSIAN_BLUR_CONFIG_SIGMA = AUGMENT_CONFIG_VARIABLES_2

#RANDOM INVERT OPTIONS
AUGMENT_RANDOM_INVERT_CONFIG_PERCENT = AUGMENT_CONFIG_VARIABLES_1