# train_personal_color12_weightlog.py
import os
import torch
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ✅ 경로 설정
DATA_DIR = 'data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 전체 데이터셋 로딩
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names = full_dataset.classes
class_to_idx = full_dataset.class_to_idx

print("✅ 클래스 목록:", class_names)

# ✅ class_to_idx 확인
print("\n🔍 class_to_idx (모델 기준 인덱스):")
for name, idx in class_to_idx.items():
    print(f" - {idx:2d}: {name}")

# ✅ 전체 데이터 중 80%를 학습 데이터로 사용한다고 가정하고, 샘플링
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
indices = torch.randperm(dataset_size)[:train_size]
labels = [full_dataset[i][1] for i in indices]

# ✅ 클래스 가중치 계산
unique_labels = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ✅ 클래스 가중치 출력
print("\n📊 클래스 가중치 매핑 (index 기반):")
for idx, weight in zip(unique_labels, class_weights):
    class_name = class_names[idx]
    print(f" - {idx:2d} ({class_name}): {weight:.4f}")
