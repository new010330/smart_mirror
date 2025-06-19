import cv2

# OpenCV 얼굴 인식용 haarcascade 로딩(최초 1회)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_face_present(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

def is_frontal_face(image):
    # 실제 서비스시 dlib/mediapipe로 각도 체크 권장. 여기서는 True로 고정
    return True
