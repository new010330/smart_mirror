from django.urls import path
from .views import analyze_faces_and_acne_level  # 또는 analyze_faces 등 실제 함수명

urlpatterns = [
    path('', analyze_faces_and_acne_level),  # 실제 view 함수에 맞게 수정
]
