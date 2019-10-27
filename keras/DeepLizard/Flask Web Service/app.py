import base64
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, jsonify

app = Flask(__name__)


def get_model():
    global model
    model = load_model('guitarSax.h5')
    print(" Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    print(type(image))
    print(type(img_to_array(image)))
    image = img_to_array(image)
    print(type(image))
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    print(type(processed_image))
    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'guitar': prediction[0][0],
            'saxophone': prediction[0][1]
        }
    }
    return jsonify(response)


@app.route('/hello')
def running():
    return "Hello World"


@app.route('/greet', methods=['POST'])
def greet():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)
