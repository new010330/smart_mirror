# skin_status/face_analysis/apps.py
from django.apps import AppConfig

class FaceAnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'skin_status.face_analysis'  # ✅ 정확히 이 경로여야 함
