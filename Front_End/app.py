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
    return SFVI.captureAction()

@app.route('/sfviSetProgramFileName', methods = ['POST'])
def sfviSetProgramFileName():
    data = request.json('param')
    print(data) #CORREGIR
    SFVI.status.setProgramFileName(data)
    return f'El valor del parámetro es: {data}'


#EJEMPLO - USANDO JS
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json
    data = data[0]['url']
    #SFVI.status.setProgramFileName(data) #REVISAR ACA
    return str(1)
    #return jsonify(message="Datos recibidos correctamente", data=data)