import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ✅ 경로 설정
DATA_DIR = 'data'
SAVE_DIR = 'saved_models'
STAGE1_DIR = os.path.join(SAVE_DIR, 'stage1')
STAGE2_DIR = os.path.join(SAVE_DIR, 'stage2')
os.makedirs(STAGE1_DIR, exist_ok=True)
os.makedirs(STAGE2_DIR, exist_ok=True)

# ✅ 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 100
LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5
IMG_SIZE = 224

# ✅ GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", DEVICE)

# ✅ 데이터 전처리
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ 데이터 로딩
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names = full_dataset.classes
num_classes = len(class_names)
with open(os.path.join(SAVE_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
    f.writelines(name + "\n" for name in class_names)
print("✅ 클래스 목록 저장 완료:", class_names)

# ✅ 훈련/검증 분리
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# ✅ oversampling을 위한 sampler 생성
autumn_classes = ['autumn_deep', 'autumn_soft', 'autumn_warm']
autumn_indices = [i for i, name in enumerate(class_names) if name in autumn_classes]
train_labels = [full_dataset[i][1] for i in train_dataset.indices]
sample_weights = [2.0 if label in autumn_indices else 1.0 for label in train_labels]
sample_weights = torch.DoubleTensor(sample_weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ 클래스 가중치 계산
labels = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ✅ 모델 생성
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

# ✅ 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR_STAGE1)

# ✅ 학습 함수 정의
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# ✅ 1단계 학습
print("\n🚀 1단계 학습 시작")
best_val_acc = 0
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"[{epoch+1:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(STAGE1_DIR, 'best_model.pt'))

# ✅ 2단계 fine-tuning
print("\n🚀 2단계 학습 시작 (fine-tuning)")
optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2)
for param in model.features.parameters():
    param.requires_grad = True

best_val_acc = 0
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"[{epoch+1:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(STAGE2_DIR, 'best_model.pt'))

# ✅ 최종 모델 저장
final_path = os.path.join(SAVE_DIR, 'final_model_efficientnet.pt')
torch.save(model.state_dict(), final_path)
print(f"\n🎉 전체 학습 완료 및 모델 저장: {final_path}")
