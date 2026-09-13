from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_detect, name='upload_and_detect'),
    path('upload/', views.upload_and_detect, name='upload_and_detect_alt'),
    path('report/', views.report_view, name='report_view'),
]
