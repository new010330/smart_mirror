# face_analysis/utils.py
import cv2

def is_face_present(image):
    # 예시: haarcascade 사용 (실제 프로젝트는 dlib, mediapipe 등 사용 권장)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def is_blurry(image, threshold=4.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"[DEBUG] Laplacian Variance: {var:.2f}, Blurry: {var < threshold}")
    return var < threshold

def is_frontal_face(image):
    # 실제로는 정면 얼굴 각도 판단 라이브러리 활용
    # 예시로 "얼굴 존재"로만 단순화
    return is_face_present(image)
