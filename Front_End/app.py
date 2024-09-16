from flask import Flask, request, make_response, redirect, abort, jsonify
from flask_cors import CORS

import SFVIMain as SFVI

app = Flask(__name__)

CORS(app)

app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True

posts = []

@app.route('/')
def index():
    return '<h1>Hello World!</h1>'


@app.route('/browser/')
def browser():
    user_agent = request.headers.get('User-Agent')
    return '<p>Your browser is {}</p>'.format(user_agent)


@app.route('/user/<name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

@app.route('/response/')
def response():
    return '<h1>Bad Request</h1>', 400

@app.route('/make_response/')
def make_response_app():
    response = make_response('<h1>This document carries a cookie!</h1>')
    response.set_cookie('answer', '42')
    return response

@app.route('/redirect/')
def redir():
    return redirect('http://localhost:5000/')

###########################
#SFVI
###########################

@app.route('/sfviCaptureAction/')
def sfviCaptureAction():
    retVar = 0
    try:
        retVar = SFVI.captureAction()
    except:
        print("sfviCaptureAction Eror")
    return retVar

@app.route('/sfviSetProgramFileName', methods = ['POST'])
def sfviSetProgramFileName():
    retVar = 0
    data = request.json
    try:    
        data = data[0]['path']
        SFVI.status.setProgramFileName(data)
    except:
        print("sfviSetProgramFileName Eror")
    return str(retVar)

@app.route('/sfviProgramOnlineStatusSet', methods = ['POST'])
def sfviProgramOnlineStatusSet():
    retVar = 0
    data = request.json
    try:
        data = data[0]['programState']
        retVar = SFVI.status.programOnlineStatusSet(data)
    except:
        print("sfviProgramOnlineStatusSet Eror")
    return str(retVar)

@app.route('/sfviResetProgramCounters/')
def sfviResetProgramCounters():
    retVar = 0
    try:
        retVar = SFVI.status.resetProgramCounters()
    except:
        print("sfviResetProgramCounters Eror")
    return str(retVar)

@app.route('/sfviTriggerProgramRun', methods = ['POST'])
def sfviTriggerProgramRun():
    retVar = 0
    data = request.json
    try:    
        data = data[0]['path']
        SFVI.status.triggerProgramRun(data)
    except:
        print("sfviTriggerProgramRun Eror")
    return str(retVar)

@app.route('/sfviGetSavedProgramsPathsFromDirectory/')
def sfviGetSavedProgramsPathsFromDirectory():
    retVar = 0
    try:
        retVar = SFVI.getSavedProgramsPathsFromDirectory()
    except:
        print("sfviGetSavedProgramsPathsFromDirectory Eror")
    return retVar

@app.route('/sfviCheckProgramType', methods = ['POST'])
def sfviTriggerProgramRun():
    retVar = 0
    data = request.json
    try:    
        data = data[0]['path']
        retVar = SFVI.checkProgramType(data)
    except:
        print("sfviCheckProgramType Eror")
    return str(retVar)

@app.route('/sfviLoadVisionProgram', methods = ['POST'])
def sfviLoadVisionProgram():
    retVar = 0
    data = request.json
    try:    
        data = data[0]['path']
        retVar = SFVI.status.loadVisionProgram(data)
    except:
        print("sfviLoadVisionProgram Eror")
    return str(retVar)

@app.route('/sfviLoadDeepLearningVisionProgram', methods = ['POST'])
def sfviLoadVisionProgram():
    retVar = 0
    data = request.json
    try:    
        data = data[0]['path']
        retVar = SFVI.status.loadDeepLearningVisionProgram(data)
    except:
        print("sfviLoadDeepLearningVisionProgram Eror")
    return str(retVar)

@app.route('/sfviCopyProgram', methods = ['POST'])
def sfviCopyProgram():
    retVar = 0
    data = request.json
    try:    
        file = data[0]['path']
        copyPath = data[0]['copyPath']
        retVar = SFVI.copyProgram(file, copyPath)
    except:
        print("sfviCopyProgram Eror")
    return str(retVar)

@app.route('/sfviDeleteProgram', methods = ['POST'])
def sfviDeleteProgram():
    retVar = 0
    data = request.json
    try:    
        file = data[0]['path']
        retVar = SFVI.deleteProgram(file)
    except:
        print("sfviDeleteProgram Eror")
    return str(retVar)

@app.route('/sfviLaunchNewVisionProgram')
def sfviLaunchNewVisionProgram():
    retVar = 0
    try:    
        retVar = SFVI.status.launchNewVisionProgram()
    except:
        print("sfviLaunchNewVisionProgram Eror")
    return str(retVar)


#EJEMPLO - USANDO JS
@app.route('/data', methods=['POST'])
def receive_data():
    retVar = 0
    try:
        retVar = SFVI.getSavedProgramsPathsFromDirectory()
    except:
        print("sfviGetSavedProgramsPathsFromDirectory Eror")
    return retVar