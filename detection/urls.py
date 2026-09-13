from django.urls import path
from . import views

urlpatterns = [
    # Map root and /upload/ to upload_and_detect view
    path('', views.upload_and_detect, name='upload_and_detect'),
    path('upload/', views.upload_and_detect, name='upload_and_detect_alt'),
]
