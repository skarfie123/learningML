from flask import Flask
app = Flask(__name__)

@app.route('/sample')
def running():
    return "Hello World"