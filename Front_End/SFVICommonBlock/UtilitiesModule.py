import os
import sys
import json
import shutil
import pathlib
import re

def searchSubfoldersForFilesEndingIn(endString, path):
    retPaths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(endString)]:
            retPaths.append(os.path.join(dirpath, filename))
    return retPaths

def searchForStringsStartingWith(startString, strings:str):
    retStrings = [string for string in strings if string.startswith(startString)]
    return retStrings

def removeFileListExtensions(files):
    retFiles = []
    for file in files:
       base, extension = os.path.splitext(file)
       retFiles.append(base)
    return retFiles

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

def saveJsonDict(path, jsonDict):
    path = path + ".json"
    with open(path, "w") as write_file:
        json.dump(jsonDict, write_file, indent=4)

def loadJsonDict(path):
    with open(path, "r") as readFile:
        jsonDict = json.load(readFile)
    return jsonDict

def isRPi():
    retVal = False
    if os.name == 'posix':
        retVal = True
    return retVal

def get_trailing_number(s):
        m = re.search(r'\d+$', s)
        return int(m.group()) if m else None

def getDirFiles(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return files