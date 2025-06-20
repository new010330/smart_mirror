import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

def main():
    train_dir = r'..\data\skin_dataset\train'
    val_dir = r'..\data\skin_dataset\val'

    batch_size = 32
    num_epochs = 25
    lr = 0.0005
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA 사용 여부:", torch.cuda.is_available())

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

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("클래스 목록:", class_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # 클래스별 가중치 계산
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("클래스별 가중치:", class_weights.tolist())

    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Loss 함수: 불균형 보정
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "mobilenet_skin_best.pth")
            print("✅ Best model saved.")

    print(f"학습 완료. 최고 검증 정확도: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
