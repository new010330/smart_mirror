# === 2. face_analysis/views.py ===

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .utils import is_face_present, is_blurry, is_frontal_face
from .level_model import predict_acne_level
from api.utils import predict_personal_color_from_path

import numpy as np
import cv2
import os
from datetime import datetime

@api_view(['POST'])
@parser_classes([MultiPartParser])
def analyze_faces_and_acne_level(request):
    images = request.FILES.getlist('image')
    if not images or len(images) != 5:
        return Response({'detail': '이미지 5장이 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
    os.makedirs(upload_dir, exist_ok=True)

    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    selected_image = None
    selected_filename = None

    for idx, img_file in enumerate(images):
        img_array = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            continue

        filename = f"{prefix}_{idx}.jpg"
        filepath = os.path.join(upload_dir, filename)
        cv2.imwrite(filepath, image)

        face_ok = is_face_present(image)
        blur_ok = not is_blurry(image)
        frontal_ok = is_frontal_face(image) if face_ok else False

        result = {
            "filename": filename,
            "face_detected": face_ok,
            "not_blurry": blur_ok,
            "frontal": frontal_ok
        }
        results.append(result)

        if selected_image is None and face_ok and blur_ok and frontal_ok:
            selected_image = image
            selected_filename = filename

    if selected_image is None:
        return Response({
            "detail": "5장 모두에서 적합한 얼굴 사진을 찾지 못했습니다.",
            "results": results
        }, status=422)

    selected_filepath = os.path.join(upload_dir, selected_filename)

    if not os.path.exists(selected_filepath):
        return Response({"error": "선택된 이미지 파일이 존재하지 않습니다."}, status=500)

    try:
        acne_level, prob = predict_acne_level(selected_image)
        print(f"[INFO] 여드름 분석 결과 - level: {acne_level}, conf: {prob}")
    except Exception as e:
        return Response({"error": f"피부 분석 오류: {str(e)}"}, status=500)

    try:
        personal_color, pc_conf = predict_personal_color_from_path(selected_filepath)
        print(f"[INFO] 퍼스널컬러 결과 - {personal_color}, conf: {pc_conf}")
    except Exception as e:
        print(f"[ERROR] personal_color 예측 실패: {str(e)}")
        personal_color, pc_conf = None, None

    return Response({
        "acne_level": int(acne_level),
        "confidence": float(prob),
        "skin_lv": "정상 (여드름 없음)" if acne_level == 0 else f"{acne_level}",
        "personal_color": personal_color if personal_color else "예측 실패",
        "pc_confidence": f"{pc_conf:.2f}%" if pc_conf is not None else "예측 실패",
    }, status=200)
