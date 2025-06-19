from django.contrib import admin
from django.urls import path, include  # include 추가 OK
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # /api/predict/ 연결 OK
    path('analyze/skin-status/', include('skin_status.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)