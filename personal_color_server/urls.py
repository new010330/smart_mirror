from django.contrib import admin
from django.urls import path, include  # include 추가 OK

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls'))  # /api/predict/ 연결 OK
]
