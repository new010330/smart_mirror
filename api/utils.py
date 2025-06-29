# api/utils.py

import cv2
import numpy as np
import os
import torch
from torchvision import models, transforms


# ✅ 모델 및 클래스 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_model_efficientnet.pt')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'class_names.txt')

# ✅ 클래스 이름 불러오기
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f]

# ✅ 모델 로드 (PyTorch)
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ✅ 경로 설정
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_model_efficientnet.pt')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'class_names.txt')

# ✅ 클래스 이름 로드
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f]

# ✅ 장치 설정 (CPU 고정)
device = torch.device("cpu")

# ✅ 모델 정의 및 로드
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ 예측 함수
def predict_personal_color_from_path(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지 로딩 실패")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img).unsqueeze(0)

    img_tensor = transform(img).unsqueeze(0).to(device)  # ✅ CPU 전송 명시

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence = float(conf.item() * 100)
    print(f"[INFO] 예측 클래스: {predicted_class}, 신뢰도: {confidence:.2f}%")
    return predicted_class, confidence
