import os
import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser

# 모델 로딩
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'personal_color_model.h5'))
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ 클래스 이름을 txt에서 동적으로 불러오기
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'class_names.txt')
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f]

class PredictColorView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file = request.FILES.get('image')
        if not file:
            return Response({"error": "이미지 파일이 없습니다."}, status=400)

        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        img = cv2.imread(file_path)
        if img is None:
            return Response({"error": "이미지를 읽을 수 없습니다."}, status=400)

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        predicted_class = class_names[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

        os.remove(file_path)

        return Response({
            "result": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })
