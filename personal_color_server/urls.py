# personal_color_server/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),              # 퍼스널컬러 예측 API
    path('api/analyze/', include('skin_status.face_analysis.urls')),  # 피부 상태 분석 API
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
