from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_hazards, name='detect_hazards'),
]
