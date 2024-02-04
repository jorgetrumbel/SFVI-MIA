import os
import sys
import json

#Paths
PATH_SAVED_PROGRAMS = ".\\savedPrograms"

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