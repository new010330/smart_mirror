# api/urls.py
from django.urls import path
from .views import PredictColorView

urlpatterns = [
    path('predict/', PredictColorView.as_view(), name='predict-color'),
]
