import tensorflow as tf
import numpy as np
import cv2
import os

# 모델과 클래스 불러오기
model = tf.keras.models.load_model('personal_color_model.h5')

# ✅ 클래스 이름은 txt에서 불러오기
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

def predict_personal_color(image_path):
    if not os.path.exists(image_path):
        return "이미지 파일이 존재하지 않음", None

    # 이미지 불러오기 및 전처리
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 예측
    pred = model.predict(img)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return predicted_class, confidence

# 테스트 실행
if __name__ == '__main__':
    test_image = 'test.jpg'
    result, confidence = predict_personal_color(test_image)
    if confidence:
        print(f"예측된 퍼스널컬러: {result} (신뢰도: {confidence:.2f}%)")
    else:
        print(result)
