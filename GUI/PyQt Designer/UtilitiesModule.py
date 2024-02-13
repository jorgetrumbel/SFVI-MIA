import os
import sys
import json
import shutil
import pathlib

def searchSubfoldersForFilesEndingIn(endString, path):
    retPaths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(endString)]:
            retPaths.append(os.path.join(dirpath, filename))
    return retPaths

def getJsonStringFromFile(path):
    with open(path, "r") as readFile:
        jsonDict = json.load(readFile)
    programString = json.dumps(jsonDict, indent = 4)
    return programString

def getFileNameFromPath(path):
     return os.path.basename(os.path.normpath(path))

def copyProgram(path, pathToCopyFile):
    path = path + ".json"
    shutil.copyfile(pathToCopyFile, path)    
    
def deleteProgram(path):
    os.remove(path)

def deleteFolderFiles(folderPath):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except:
            pass

def copyFolderFiles(folderPath, destinationFolder):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path):
                shutil.copy(file_path, destinationFolder)
        except:
            pass

def checkIfPathExists(path):
    return os.path.exists(path)

def getFileSuffix(path):
    return pathlib.Path(path).suffix