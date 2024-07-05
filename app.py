from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

@app.route('/form')
def mlq():
    return render_template('form.html')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
img_height, img_width = 224, 224
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

svm_model = joblib.load('svm_mnv2.joblib')
label_encoder = joblib.load('label_encoder.joblib')

class_names = ['Anthracnose', 'Algal leaf', 'Bird eye spot', 'Brown blight', 'Gray light', 'Healthy', 'Red leaf spot', 'White spot']

def extract_features(img):
    img_array = np.array(img.resize((img_height, img_width)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = base_model.predict(img_array)
    features = global_average_layer(features).numpy()
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        img = Image.open(file)
        features = extract_features(img)
        prediction = svm_model.predict(features)
        predicted_index = int(prediction[0])
        class_name = class_names[predicted_index]
        return jsonify({'predicted_class': class_name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
