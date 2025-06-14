import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# 📁 모델 및 클래스 파일 경로
MODEL_PATH = os.path.join('saved_models', 'personal_color_model.h5')
CLASS_PATH = os.path.join('saved_models', 'class_names.txt')

# ✅ 모델과 클래스 불러오기 함수
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        sys.exit()
    if not os.path.exists(CLASS_PATH):
        print(f"❌ 클래스 파일이 없습니다: {CLASS_PATH}")
        sys.exit()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip().replace("\\", "_") for line in f]

    if not class_names:
        print("❌ 클래스 목록이 비어 있습니다.")
        sys.exit()

    print(f"✅ 클래스 목록 로드됨: {class_names}")
    return model, class_names

# ✅ 예측 함수
def predict_personal_color(image_path, model, class_names):
    if not os.path.exists(image_path):
        return "❌ 이미지 파일이 존재하지 않음", None

    # 이미지 불러오기 (OpenCV: BGR → RGB 변환)
    img = cv2.imread(image_path)
    if img is None:
        return "❌ 이미지를 읽을 수 없음", None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 예측
    pred = model.predict(img)
    predicted_index = np.argmax(pred)
    predicted_class = class_names[predicted_index]
    confidence = np.max(pred) * 100

    return predicted_class, confidence

# ✅ 테스트 실행
if __name__ == '__main__':
    test_image = 'test9.jpeg'
    print(f"📷 예측 시작: {test_image}")

    model, class_names = load_model_and_classes()
    result, confidence = predict_personal_color(test_image, model, class_names)

    if confidence:
        print(f"✅ 예측된 퍼스널컬러: {result} (신뢰도: {confidence:.2f}%)")
    else:
        print(f"❌ 예측 실패: {result}")
