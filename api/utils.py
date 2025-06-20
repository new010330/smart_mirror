# === 1. api/utils.py ===

import cv2
import numpy as np
import os
import tensorflow as tf

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'personal_color_model.h5'))
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'class_names.txt')
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f]

def predict_personal_color_from_path(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지 로딩 실패")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)
    return predicted_class, confidence