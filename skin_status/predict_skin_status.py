import torch
from torchvision import models, transforms
from PIL import Image

# 1. 클래스 이름
class_names = ['normal', 'acne']

# 2. 모델 구조 정의 (클래스 수 = 2)
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2)

# 3. 파라미터 로딩
state_dict = torch.load('skin_status/mobilenet_skin_best.pth', map_location='cpu')
model.load_state_dict(state_dict)

# 4. 추론 모드
model.eval()

# 5. 예측 함수
def predict_skin_status(image: Image.Image) -> dict:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, 1)[0][predicted.item()]
        return {
            "result": class_names[predicted.item()],
            "confidence": float(confidence)
        }
