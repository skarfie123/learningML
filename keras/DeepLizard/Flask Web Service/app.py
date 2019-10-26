from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/hello')
def running():
    return "Hello World"

@app.route('/greet', methods=['POST'])
def greet():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting' : 'Hello, ' + name + '!'
    }
    return jsonify(response)