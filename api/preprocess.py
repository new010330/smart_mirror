import cv2
import numpy as np

def preprocess_image(file_path, target_size=(224, 224), use_face_crop=True):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("이미지 로딩 실패")

    # 선택적으로 얼굴만 크롭
    if use_face_crop:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            img = img[y:y+h, x:x+w]

    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img
