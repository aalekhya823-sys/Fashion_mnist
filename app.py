from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("fashion_model.h5")

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

def preprocess_image(image):
    image = image.convert("L")      # grayscale
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file)
        processed = preprocess_image(image)
        pred = model.predict(processed)
        prediction = class_names[np.argmax(pred)]
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)