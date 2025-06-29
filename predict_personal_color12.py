import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ✅ 경로 설정
MODEL_PATH = os.path.join('saved_models', 'final_model_efficientnet.pt')
CLASS_PATH = os.path.join('saved_models', 'class_names.txt')

# ✅ 모델 및 클래스 로드
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        exit()
    if not os.path.exists(CLASS_PATH):
        print(f"❌ 클래스 파일이 없습니다: {CLASS_PATH}")
        exit()

    model = models.efficientnet_b0(pretrained=False)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f]
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    model.to('cpu')  # ✅ 명시적으로 CPU에 할당

    print(f"✅ 클래스 목록 로드 완료: {class_names}")
    return model, class_names

# ✅ 얼굴 crop 함수 (MediaPipe 사용)
def crop_face(img):
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)
        if not results.detections:
            print("⚠️ 얼굴 미검출 - 예측 중단")
            return None

        h, w, _ = img.shape
        box = results.detections[0].location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)
        margin = int(0.2 * bh)

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin)
        y2 = min(h, y + bh + margin)
        return img[y1:y2, x1:x2]

# ✅ 전처리 보정 함수들
def apply_hist_eq(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def simple_white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# ✅ 예측 함수
def predict_personal_color(image_path, model, class_names, debug=False):
    if not os.path.exists(image_path):
        return "이미지가 없습니다.", None

    img = cv2.imread(image_path)
    if img is None:
        return "이미지를 불러올 수 없습니다.", None

    face_img = crop_face(img)
    if face_img is None:
        return "얼굴 인식 실패", None

    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = simple_white_balance(img)
    img = apply_clahe(img)
    img = apply_hist_eq(img)

    if debug:
        print(f"📐 전처리 후 이미지 크기: {img.shape}")

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img_tensor = torch.tensor(img).unsqueeze(0).to('cpu')  # ✅ CPU 명시

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        confidence = conf.item() * 100
        predicted_class = class_names[pred_idx.item()]

    if confidence < 30:
        print(f"⚠️ 신뢰도 낮음: {confidence:.2f}%")

    if debug:
        print("\n📊 클래스별 확률:")
        for i, class_name in enumerate(class_names):
            print(f" - {class_name}: {probs[0][i]*100:.2f}%")

    return predicted_class, confidence

# ✅ 실행부 테스트
if __name__ == '__main__':
    test_image = 'test33.jpg'
    print(f"📷 예측 시작: {test_image}")
    model, class_names = load_model_and_classes()
    result, confidence = predict_personal_color(test_image, model, class_names, debug=True)

    if confidence is not None:
        print(f"✅ 예측 결과: {result} (신뢰도: {confidence:.2f}%)")
    else:
        print(f"❌ 예측 실패: {result}")
