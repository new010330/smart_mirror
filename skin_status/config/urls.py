from django.contrib import admin
from django.urls import path
from face_analysis.views import analyze_faces

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/analyze/', analyze_faces),  # 요청 경로 맞춤
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
