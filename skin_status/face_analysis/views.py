from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .utils import is_face_present, is_frontal_face
from .level_model import predict_acne_level
from api.utils import predict_personal_color_from_path

import numpy as np
import cv2
import os
from datetime import datetime
import mediapipe as mp

# mediapipe 얼굴 탐지 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def crop_face_from_image(image: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if not results.detections:
        print(f"[WARNING] 얼굴 탐지 실패")
        raise ValueError("얼굴을 찾지 못했습니다.")

    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    h, w, _ = image.shape

    x = max(int(bboxC.xmin * w), 0)
    y = max(int(bboxC.ymin * h), 0)
    x2 = min(int((bboxC.xmin + bboxC.width) * w), w)
    y2 = min(int((bboxC.ymin + bboxC.height) * h), h)

    return image[y:y2, x:x2]

@api_view(['POST'])
@parser_classes([MultiPartParser])
def analyze_faces_and_acne_level(request):
    images = request.FILES.getlist('image')[-5:]
    if len(images) < 5:
        return Response({'detail': '이미지 5장이 필요합니다.'}, status=400)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
    os.makedirs(upload_dir, exist_ok=True)

    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    analysis_inputs = []

    for idx, img_file in enumerate(images):
        try:
            img_array = np.frombuffer(img_file.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] 이미지 디코딩 실패: {idx}")
                continue

            print(f"[INFO] 디코딩 성공 - idx {idx}, shape={image.shape}")

            filename = f"{prefix}_{idx}.jpg"
            filepath = os.path.join(upload_dir, filename)
            cv2.imwrite(filepath, image)

            face_ok = is_face_present(image)
            frontal_ok = is_frontal_face(image) if face_ok else False

            print(f"[INFO] 얼굴 탐지 결과 - idx {idx}: face={face_ok}, frontal={frontal_ok}")

            results.append({
                "filename": filename,
                "face_detected": face_ok,
                "frontal": frontal_ok
            })

            if face_ok and frontal_ok:
                try:
                    face_crop = crop_face_from_image(image)
                    print(f"[INFO] 얼굴 추출 성공 - idx {idx}, crop shape={face_crop.shape}")
                    analysis_inputs.append((face_crop, filepath))
                except Exception as e:
                    print(f"[WARN] 얼굴 추출 실패 - idx {idx}: {str(e)}")
        except Exception as e:
            print(f"[ERROR] 이미지 처리 중 에러 발생 - idx {idx}: {str(e)}")

    if not analysis_inputs:
        return Response({
            "detail": "적합한 얼굴 사진이 없습니다.",
            "results": results
        }, status=422)

    best_level = None
    best_conf = 0.0
    best_path = None

    for idx, (face_image, filepath) in enumerate(analysis_inputs):
        try:
            level, prob = predict_acne_level(face_image)
            print(f"[INFO] 분석 결과 - idx {idx}: level={level}, conf={prob}")

            if best_level is None or level < best_level or (level == best_level and prob > best_conf):
                best_level = level
                best_conf = prob
                best_path = filepath
        except Exception as e:
            print(f"[ERROR] 여드름 분석 실패 - idx {idx}: {str(e)}")

    try:
        personal_color, pc_conf = predict_personal_color_from_path(best_path)
        print(f"[INFO] 퍼스널컬러: {personal_color}, conf: {pc_conf}")
    except Exception as e:
        print(f"[ERROR] 퍼스널컬러 예측 실패: {str(e)}")
        personal_color, pc_conf = None, None

    try:
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))
    except Exception as e:
        print(f"[WARN] 이미지 삭제 실패: {str(e)}")

    return Response({
        "acne_level": int(best_level),
        "confidence": float(best_conf),
        "skin_lv": "정상 (여드름 없음)" if best_level == 0 else f"{best_level}",
        "personal_color": personal_color if personal_color else "예측 실패",
        "pc_confidence": f"{pc_conf:.2f}%" if pc_conf is not None else "예측 실패",
        "details": results
    }, status=200)