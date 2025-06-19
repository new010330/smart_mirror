from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .utils import is_face_present, is_blurry, is_frontal_face

import numpy as np
import cv2
import os
from datetime import datetime

@api_view(['POST'])
@parser_classes([MultiPartParser])
def analyze_faces(request):
    images = request.FILES.getlist('image')  # 프론트에서 'image'로 보냄
    if not images or len(images) != 5:
        return Response({'detail': '이미지 5장이 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
    os.makedirs(upload_dir, exist_ok=True)

    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    valid_face_count = 0
    results = []

    for idx, img_file in enumerate(images):
        img_array = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 파일 저장
        filename = f"{prefix}_{idx}.jpg"
        filepath = os.path.join(upload_dir, filename)
        cv2.imwrite(filepath, image)

        # 판별
        face_ok = is_face_present(image)
        blur_ok = not is_blurry(image)
        frontal_ok = is_frontal_face(image) if face_ok else False

        # 로그 출력
        blur_status = "선명" if blur_ok else "흐림"
        face_status = "얼굴" if face_ok else "얼굴없음"
        frontal_status = "정면" if frontal_ok else "비정면"
        print(f"[이미지 저장] 파일명: {filename}, 판별: {blur_status}, {face_status}, {frontal_status}")

        result = {
            "filename": filename,
            "face_detected": face_ok,
            "not_blurry": blur_ok,
            "frontal": frontal_ok
        }
        results.append(result)

        if face_ok and blur_ok and frontal_ok:
            valid_face_count += 1

    if valid_face_count == 0:
        return Response({
            "detail": "5장 모두에서 적합한 얼굴 사진을 찾지 못했습니다.",
            "results": results
        }, status=422)

    return Response({
        "detail": "적어도 한 장 이상의 유효한 얼굴 사진이 있습니다.",
        "results": results
    }, status=200)
