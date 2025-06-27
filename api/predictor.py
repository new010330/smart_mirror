# api/predictor.py

import torch
import cv2
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image

# 모델 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_model_efficientnet.pt')
CLASS_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'class_names.txt')

model = models.efficientnet_b0(pretrained=False)
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]
model.classifier[1].out_features = len(class_names)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def predict_personal_color_from_path(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지 로딩 실패")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        confidence = conf.item() * 100
        predicted_class = class_names[pred_idx.item()]
    return predicted_class, confidence
