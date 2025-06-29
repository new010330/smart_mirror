# face_analysis/utils.py

import cv2
import mediapipe as mp

# 얼굴 감지 모델 설정
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def is_face_present(image):
    """
    얼굴이 존재하는지 확인
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)
    return results.detections is not None and len(results.detections) > 0


def is_frontal_face(image):
    """
    정면 얼굴 여부 확인
    눈과 코 keypoints 기준으로 정면 여부 판단
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)

    if not results.detections:
        return False

    keypoints = results.detections[0].location_data.relative_keypoints
    if len(keypoints) < 3:
        return False

    left_eye = keypoints[0]
    right_eye = keypoints[1]
    nose_tip = keypoints[2]

    eye_symmetry = abs(left_eye.x - (1 - right_eye.x)) < 0.2
    nose_centered = abs(nose_tip.x - 0.5) < 0.1

    return eye_symmetry and nose_centered


def is_blurry(image, threshold=4.5):
    """
    이미지가 흐린지 여부 판단
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"[DEBUG] Laplacian Variance: {var:.2f}, Blurry: {var < threshold}")
    return var < threshold
