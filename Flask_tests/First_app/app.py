from flask import Flask, request, make_response, redirect, abort

app = Flask(__name__)

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

'''
@app.route('/userid/<id>/')
def get_user(id):
    user = load_user(id)
    if user is 0:
        abort(404)
    return '<h1>Hello, {}</h1>'.format(user.name)

def load_user(id):
    if id is '0':
        return 0
    else:
        return id
'''