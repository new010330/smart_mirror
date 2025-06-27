# === 1. api/utils.py ===

import cv2
import numpy as np
import os
import torch
from torchvision import models, transforms

# ✅ 모델 및 클래스 경로 설정
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_model_efficientnet.pt'))
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'class_names.txt')

# ✅ 클래스 이름 불러오기
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f]

# ✅ 모델 로드 (PyTorch)
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ 예측 함수
def predict_personal_color_from_path(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지 로딩 실패")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence = float(conf.item() * 100)

    return predicted_class, confidence
