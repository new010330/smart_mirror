# skin_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # ================================
    # 1. 하이퍼파라미터 및 환경 설정
    # ================================
    # 실제 데이터 경로를 아래와 같이 맞춰준다
    train_dir = r'C:\junho\web\data\skin_condition\open_data\data\Training\source_data'
    val_dir   = r'C:\junho\web\data\skin_condition\open_data\data\Validation\source_data'

    batch_size = 32
    num_epochs = 25
    lr = 0.0005
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================
    # 2. 데이터 전처리 및 데이터로더
    # ================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 클래스 자동 추출
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("훈련 클래스 목록:", class_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    # ================================
    # 3. MobileNetV2 모델 생성 및 커스터마이즈
    # ================================
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # ================================
    # 4. 손실함수, 옵티마이저
    # ================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ================================
    # 5. 학습 및 검증 루프
    # ================================
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
        train_acc = running_corrects / total

        # 검증
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
        val_acc = val_corrects / val_total

        print(f"[Epoch {epoch+1:02d}/{num_epochs}] "
              f"Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

        # 베스트 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "mobilenet_skin_best.pth")
            print("Best model saved.")

    print("학습 완료. 최종 최고 검증 정확도: {:.4f}".format(best_val_acc))

if __name__ == '__main__':
    main()
