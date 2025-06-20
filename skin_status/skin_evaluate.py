import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 설정 ===
test_dir = r'..\data\skin_dataset\val'
model_path = 'mobilenet_skin_best.pth'
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 전처리 ===
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === 데이터셋 ===
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# === 모델 로드 ===
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === 평가 ===
y_true, y_pred, y_probs, img_paths = [], [], [], []

start_time = time.time()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        outputs = model(inputs)
        probs = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

        batch_indices = list(range(len(img_paths), len(img_paths) + len(labels)))
        img_paths.extend([test_dataset.samples[idx][0] for idx in batch_indices])

end_time = time.time()

# === 결과 출력 ===
total_time = end_time - start_time
avg_time = total_time / len(test_dataset)
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print(f"\n== 평가 결과 ==")
print(f"전체 이미지 수: {len(test_dataset)}")
print(f"정확도(Accuracy): {accuracy:.4f}")
print(f"전체 소요 시간: {total_time:.2f}초")
print(f"평균 추론 시간: {avg_time:.3f}초")

# === 리포트 ===
print("\n== 클래스별 정밀도 리포트 ==")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n== 혼동 행렬 (Confusion Matrix) ==")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# === 혼동 행렬 시각화 ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')  # 저장
plt.show()

# === 상세 저장 ===
with open('skin_eval_detail.txt', 'w', encoding='utf-8') as f:
    f.write("img_path\tlabel\tpred\tprob\n")
    for path, true, pred, probs in zip(img_paths, y_true, y_pred, y_probs):
        prob_pred = probs[pred]
        f.write(f"{path}\t{class_names[true]}\t{class_names[pred]}\t{prob_pred:.3f}\n")

print("\n(상세 결과는 skin_eval_detail.txt 및 confusion_matrix.png 참고)")
