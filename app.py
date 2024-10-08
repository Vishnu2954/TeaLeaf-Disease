from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import logging
import time
import uvicorn

app = FastAPI()

@app.get("/form")
async def form():
    with open("form.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

img_height, img_width = 224, 224
base_model = tf.keras.applications.EfficientNetV2B1(input_shape=(img_height, img_width, 3),
                                                    include_top=False,
                                                    pooling='avg',
                                                    classifier_activation='softmax',
                                                    weights='imagenet')
base_model.trainable = False

svm_model = joblib.load('svm_env2.joblib')

class_names = ['Anthracnose', 'Algal leaf', 'Bird eye spot', 'Brown blight', 'Gray light', 'Red leaf spot',
               'White spot']

logging.basicConfig(level=logging.INFO)


def extract_features(img):
    start_time = time.time()
    img_array = np.array(img.resize((img_height, img_width)))
    img_array = np.expand_dims(img_array, axis=0)
    features_start_time = time.time()
    features = base_model.predict(img_array)
    end_time = time.time()
    logging.info(f"Image preprocessing time: {features_start_time - start_time} seconds")
    logging.info(f"Feature extraction time: {end_time - features_start_time} seconds")
    logging.info(f"Total extraction time: {end_time - start_time} seconds")
    return features


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        logging.info("Opening image")
        img = Image.open(file.file)
        logging.info("Extracting features")
        features = extract_features(img)
        logging.info("Making prediction")
        prediction = svm_model.predict(features)
        class_name = class_names[prediction[0]]
        logging.info(f"Prediction complete: {class_name}")
        return {'predicted_class': class_name}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=8000)
