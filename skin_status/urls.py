# skin_status/urls.py
from django.urls import path
from .views import SkinStatusView

urlpatterns = [
    path('', SkinStatusView.as_view(), name='skin-status'),
]
